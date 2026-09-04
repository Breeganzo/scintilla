"""Grounded answering: build a prompt from retrieved papers, and check what comes back.

**The claim being defended.** A retrieval-augmented answer is only worth more
than a plain model answer if the text is actually tied to the retrieved
documents. That is an empirical property, not a design intention, so this module
is written so it can be checked: every passage handed to the model is numbered,
every citation the model emits is parsed back out, and any citation pointing at
a number that was never supplied is counted as a fabrication.

**Why abstention is a first-class output rather than a phrase to grep for.**
The golden set contains ten unanswerable queries - protein folding, CRISPR
off-target effects, mortgage default risk, lease review - placed deliberately
next to a corpus of physics and information-retrieval papers, because those are
the queries where a confident wrong answer is most plausible and most damaging:
retrieval always returns *something*, and a model handed five irrelevant
abstracts will happily write around them. Detecting refusal by looking for
"I don't know"
in free text is unreliable in both directions: a model can refuse in wording
the check does not recognise, and can also answer a question while mentioning
that some detail is unknown. So the prompt requires an exact sentinel on its
own first line, and abstention is defined as emitting it. That turns a fuzzy
judgement into a decidable one, which is the only way to put a number on it.

**Why the sentinel is checked at the start and not anywhere.** A model that
answers at length and then adds "otherwise INSUFFICIENT_CONTEXT" has not
abstained. Requiring the token to open the response makes the distinction
unambiguous and makes an ambiguous response count as an answer - the
conservative direction, since it means the abstention rate can only be
understated, never inflated.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from papers.models import Paper
from search.llm import Completion, LLMError, LLMProvider, get_provider
from search.retrievers.base import RetrievalResult

logger = logging.getLogger(__name__)

# The exact string the model must emit, alone on the first line, to refuse.
# Upper case with an underscore because it is a token, not prose: it is
# unlikely to appear by accident in an answer about particle physics, and it is
# visually obvious in a transcript.
#
# noqa S105: bandit flags any constant whose name contains "TOKEN" as a possible
# hardcoded credential. This one is a sentinel in a prompt, not a secret.
ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"  # noqa: S105

# How many retrieved papers are put in the prompt. Ten results are returned by
# search, but abstracts are long and every extra passage both costs tokens and
# gives the model one more thing to cite loosely. Five is enough that a correct
# answer is usually reachable and few enough that reading the prompt by hand is
# still practical when a bad answer needs diagnosing.
DEFAULT_CONTEXT_SIZE = 5

# Abstracts are already short; this only guards against a pathological one.
MAX_PASSAGE_CHARS = 2000

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = f"""You answer questions using only the numbered passages supplied to you.

Rules, in priority order:
1. If the passages do not contain enough information to answer the question, \
your entire reply must be exactly `{ABSTAIN_TOKEN}` on the first line, and nothing else. \
Do not apologise, explain, or offer general knowledge.
2. Never use information that is not in the passages, even if you are confident it is true.
3. Cite every claim with the passage number in square brackets, like [2]. \
A sentence drawn from two passages gets both, like [1][3].
4. Only cite numbers that were actually supplied to you.
5. Be concise. Three or four sentences is usually enough.

The passages are abstracts of academic papers. They are retrieved by relevance, \
so some of them may be irrelevant to the question; ignore those rather than \
stretching them to fit."""


@dataclass(frozen=True)
class Passage:
    """One numbered piece of context, as the model will see it."""

    number: int
    arxiv_id: str
    title: str
    text: str

    def render(self) -> str:
        return f"[{self.number}] {self.title}\n{self.text}"


@dataclass(frozen=True)
class AnswerResult:
    """An answer, and everything needed to judge whether to trust it."""

    query: str
    text: str
    abstained: bool
    passages: list[Passage]
    citations: list[int]
    invalid_citations: list[int]
    degraded: bool
    model: str
    elapsed_ms: int
    reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def cited_ids(self) -> list[str]:
        """The arXiv IDs the answer actually pointed at, in citation order."""
        by_number = {p.number: p.arxiv_id for p in self.passages}
        seen: list[str] = []
        for number in self.citations:
            arxiv_id = by_number.get(number)
            if arxiv_id and arxiv_id not in seen:
                seen.append(arxiv_id)
        return seen

    @property
    def citation_precision(self) -> float | None:
        """Fraction of citations that referred to a supplied passage.

        ``None`` when the answer cited nothing at all, because 0/0 is not a
        score of zero - an abstention cites nothing and should not be recorded
        as perfectly wrong. Callers average over the answers where this is a
        number and count the rest separately.
        """
        if not self.citations:
            return None
        valid = len(self.citations) - len(self.invalid_citations)
        return valid / len(self.citations)

    @property
    def is_grounded(self) -> bool:
        """Cited at least once, and every citation resolved."""
        return bool(self.citations) and not self.invalid_citations


def build_passages(
    results: list[RetrievalResult],
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> list[Passage]:
    """Turn retrieval results into numbered passages, preserving rank order.

    Papers are fetched in one query and reordered in Python rather than looked
    up one at a time, because ``filter(arxiv_id__in=...)`` returns rows in
    whatever order the database finds convenient - and passage numbers must
    follow retrieval rank, or citation [1] would not mean the top result.
    """
    top = results[:context_size]
    if not top:
        return []

    papers = Paper.objects.filter(arxiv_id__in=[r.arxiv_id for r in top]).only(
        "arxiv_id", "title", "abstract"
    )
    by_id = {p.arxiv_id: p for p in papers}

    passages: list[Passage] = []
    for result in top:
        paper = by_id.get(result.arxiv_id)
        if paper is None:
            # The index knows about a paper the database does not. Skipping is
            # right: a passage cannot be built, and inventing a placeholder
            # would give the model a citable number backed by nothing.
            logger.warning("Retrieved %s is missing from the database", result.arxiv_id)
            continue
        passages.append(
            Passage(
                number=len(passages) + 1,
                arxiv_id=paper.arxiv_id,
                title=paper.title.strip(),
                text=paper.abstract.strip()[:MAX_PASSAGE_CHARS],
            )
        )
    return passages


def build_user_prompt(query: str, passages: list[Passage]) -> str:
    """The user turn: passages first, question last.

    The question goes at the end because instructions placed after a long
    context are attended to more reliably than ones buried before it - the same
    reason the abstention rule is restated here rather than left in the system
    prompt alone.
    """
    body = "\n\n".join(p.render() for p in passages)
    return (
        f"Passages:\n\n{body}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the passages above, citing them by number. "
        f"If they are insufficient, reply with exactly {ABSTAIN_TOKEN}."
    )


def parse_citations(text: str, valid_numbers: set[int]) -> tuple[list[int], list[int]]:
    """Extract cited numbers in order, and which of them were never supplied.

    Duplicates are kept. If a model cites [7] three times when only five
    passages exist, that is three fabricated references, and collapsing them to
    one would flatter the precision figure.
    """
    cited = [int(m.group(1)) for m in CITATION_PATTERN.finditer(text)]
    invalid = [n for n in cited if n not in valid_numbers]
    return cited, invalid


def is_abstention(text: str) -> bool:
    """True when the reply opens with the sentinel."""
    stripped = text.strip()
    if not stripped:
        return False
    first_line = stripped.splitlines()[0].strip().strip("`*. ")
    return first_line.upper() == ABSTAIN_TOKEN


def answer(
    query: str,
    results: list[RetrievalResult],
    provider: LLMProvider | None = None,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> AnswerResult:
    """Answer ``query`` from ``results``, or explain why there is no answer.

    Never raises for an unavailable or failing provider. Search has already
    succeeded by this point, and losing those results because an optional
    service is down would be a worse outcome than returning them without
    prose - but ``degraded`` is set so the caller cannot mistake silence for
    a refusal.
    """
    started = time.perf_counter()
    provider = provider if provider is not None else get_provider()
    passages = build_passages(results, context_size=context_size)

    def degraded(reason: str, model: str = "") -> AnswerResult:
        return AnswerResult(
            query=query,
            text="",
            abstained=False,
            passages=passages,
            citations=[],
            invalid_citations=[],
            degraded=True,
            model=model,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            reason=reason,
        )

    if not passages:
        # Retrieval found nothing, so there is nothing to be grounded in.
        # Calling the model here is exactly how an ungrounded answer gets
        # produced, so the layer refuses on the system's behalf.
        return AnswerResult(
            query=query,
            text="",
            abstained=True,
            passages=[],
            citations=[],
            invalid_citations=[],
            degraded=False,
            model="",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            reason="no results to ground an answer in",
        )

    if not provider.available:
        return degraded("no LLM provider configured")

    try:
        completion: Completion = provider.complete(
            SYSTEM_PROMPT, build_user_prompt(query, passages)
        )
    except LLMError as exc:
        logger.warning("Answering degraded: %s", exc)
        return degraded(str(exc), model=getattr(provider, "model", ""))

    text = completion.text.strip()
    abstained = is_abstention(text)
    valid_numbers = {p.number for p in passages}
    citations, invalid = ([], []) if abstained else parse_citations(text, valid_numbers)

    return AnswerResult(
        query=query,
        text=text,
        abstained=abstained,
        passages=passages,
        citations=citations,
        invalid_citations=invalid,
        degraded=False,
        model=completion.model,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        usage={
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        },
    )


__all__ = [
    "ABSTAIN_TOKEN",
    "DEFAULT_CONTEXT_SIZE",
    "SYSTEM_PROMPT",
    "AnswerResult",
    "Passage",
    "answer",
    "build_passages",
    "build_user_prompt",
    "is_abstention",
    "parse_citations",
]
