"""Measuring the answering layer, not just the retrieval under it.

Day 9 measured whether the right documents come back. That is necessary and not
sufficient: a system can retrieve perfectly and still fabricate, and it can
retrieve badly and still be safe if it refuses. This module measures the two
properties that separate those cases.

**Abstention rate on unanswerable queries** - of the ten golden queries whose
answers are not in the corpus, how many did the system decline? This is the
number that most RAG demos never produce, because their evaluation set contains
only questions they can answer. Retrieval always returns ten results; the
question is whether the layer above them notices that all ten are irrelevant.

**False abstention rate on answerable queries** - abstaining on everything
would score 100% on the metric above, so it has to be reported against its
opposite. The pair is the point: either number alone can be gamed by a
degenerate system, and both together cannot.

**Citation precision** - of the passage numbers the model emitted, how many
referred to a passage that was actually supplied? A citation to [7] when six
passages were given is a hallucinated reference, and it is detectable
mechanically, which is why it is measured here. What is *not* measured is
whether the cited passage genuinely supports the sentence: that needs a human
or a judge model, and claiming it without one would be exactly the kind of
unearned assertion this project exists to avoid. The README says so explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evaluation.golden import GoldenQuery, GoldenSet, load_golden_set
from evaluation.harness import WARMUP_QUERY
from search.answering import DEFAULT_CONTEXT_SIZE, AnswerResult, answer
from search.llm import LLMProvider, get_provider
from search.retrievers import get_retriever
from search.retrievers.base import DEFAULT_TOP_K, Retriever


@dataclass(frozen=True)
class AnswerOutcome:
    """One golden query, put through retrieval and then through the model."""

    query_id: str
    query_class: str
    query: str
    answerable: bool
    result: AnswerResult
    retrieved_ids: list[str]
    relevant_ids: list[str]

    @property
    def abstained(self) -> bool:
        return self.result.abstained

    @property
    def correct(self) -> bool:
        """Did the system do the right thing at the abstention level?

        Deliberately shallow: for an unanswerable query the right thing is to
        refuse, and for an answerable one it is to attempt an answer. This says
        nothing about whether an attempted answer is *good* - that would need
        judged relevance, which does not exist here.
        """
        return self.abstained if not self.answerable else not self.abstained

    @property
    def cited_relevant(self) -> int:
        """How many of the cited papers were labelled relevant."""
        relevant = set(self.relevant_ids)
        return sum(1 for arxiv_id in self.result.cited_ids if arxiv_id in relevant)

    @property
    def citation_grounding(self) -> float | None:
        """Fraction of cited papers that the golden set labels relevant.

        A stricter check than :attr:`AnswerResult.citation_precision`: that one
        asks whether a citation pointed at a real passage, this one asks
        whether the passage was actually the right paper. ``None`` when nothing
        was cited, for the same 0/0 reason as everywhere else in this project.
        """
        cited = self.result.cited_ids
        if not cited:
            return None
        return self.cited_relevant / len(cited)


@dataclass(frozen=True)
class AnswerReport:
    """Everything measured in one sweep of the golden set."""

    mode: str
    model: str
    top_k: int
    context_size: int
    golden_set_version: int
    outcomes: list[AnswerOutcome]
    degraded: bool = False
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def answerable(self) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if o.answerable]

    @property
    def unanswerable(self) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if not o.answerable]

    @property
    def abstention_rate(self) -> float:
        """Share of unanswerable queries the system refused. Higher is better."""
        pool = self.unanswerable
        return sum(o.abstained for o in pool) / len(pool) if pool else 0.0

    @property
    def false_abstention_rate(self) -> float:
        """Share of answerable queries the system refused. Lower is better."""
        pool = self.answerable
        return sum(o.abstained for o in pool) / len(pool) if pool else 0.0

    @property
    def answered(self) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if not o.abstained and not o.result.degraded]

    @property
    def citation_precision(self) -> float | None:
        """Macro-average over answers that cited anything."""
        scores = [
            o.result.citation_precision
            for o in self.answered
            if o.result.citation_precision is not None
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def citation_grounding(self) -> float | None:
        scores = [o.citation_grounding for o in self.answerable if o.citation_grounding is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def uncited_answers(self) -> int:
        """Answers that were produced with no citation at all.

        Not an error the metric above can see - citation precision is undefined
        for them, so they would silently vanish from the average. An uncited
        answer is the least grounded thing the system can emit, so it is
        counted separately rather than dropped.
        """
        return sum(1 for o in self.answered if not o.result.citations)

    @property
    def hallucinated_citations(self) -> int:
        return sum(len(o.result.invalid_citations) for o in self.outcomes)

    @property
    def median_latency_ms(self) -> int:
        times = sorted(o.result.elapsed_ms for o in self.outcomes)
        return times[len(times) // 2] if times else 0


def run_answer_query(
    golden: GoldenQuery,
    retriever: Retriever,
    provider: LLMProvider,
    top_k: int = DEFAULT_TOP_K,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> AnswerOutcome:
    """Retrieve for one golden query, then answer from what came back."""
    results = retriever.retrieve(golden.query, top_k=top_k)
    return AnswerOutcome(
        query_id=golden.id,
        query_class=golden.query_class,
        query=golden.query,
        answerable=not golden.is_unanswerable,
        result=answer(golden.query, results, provider=provider, context_size=context_size),
        retrieved_ids=[r.arxiv_id for r in results],
        relevant_ids=list(golden.relevant_ids),
    )


def run_answer_eval(
    mode: str = "dense",
    golden: GoldenSet | None = None,
    top_k: int = DEFAULT_TOP_K,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    provider: LLMProvider | None = None,
    retriever: Retriever | None = None,
    warm_up: bool = True,
    pause_seconds: float = 0.0,
) -> AnswerReport:
    """Sweep the whole golden set through retrieval and answering.

    ``mode`` defaults to ``dense`` rather than ``hybrid`` because Day 9's
    ablation found no evidence hybrid retrieves better, and using the weaker
    configuration to feed the answering layer would confound two questions.

    ``pause_seconds`` exists for hosted-API rate limits. Free tiers cap requests
    per minute, and a 429 in the middle of a sweep produces a report whose
    abstention rate is really a measure of the rate limiter.
    """
    golden = golden or load_golden_set()
    retriever = retriever if retriever is not None else get_retriever(mode)
    provider = provider if provider is not None else get_provider()

    if warm_up:
        # Same reason as the retrieval harness: the first dense query pays a
        # multi-second cost to load the embedding model, and charging that to
        # whichever query happens to be first makes the latency column a lie.
        retriever.retrieve(WARMUP_QUERY, top_k=1)

    outcomes: list[AnswerOutcome] = []
    for index, query in enumerate(golden.queries):
        if pause_seconds and index:
            time.sleep(pause_seconds)
        outcomes.append(
            run_answer_query(
                query,
                retriever=retriever,
                provider=provider,
                top_k=top_k,
                context_size=context_size,
            )
        )

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for outcome in outcomes:
        for key in usage:
            usage[key] += outcome.result.usage.get(key, 0)

    models = {o.result.model for o in outcomes if o.result.model}
    return AnswerReport(
        mode=getattr(retriever, "name", mode),
        model=next(iter(models), "") if len(models) == 1 else ",".join(sorted(models)),
        top_k=top_k,
        context_size=context_size,
        golden_set_version=golden.version,
        outcomes=outcomes,
        degraded=any(o.result.degraded for o in outcomes),
        usage=usage,
    )


__all__ = [
    "AnswerOutcome",
    "AnswerReport",
    "run_answer_eval",
    "run_answer_query",
]
