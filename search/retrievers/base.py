"""The retrieval interface, and why it is a Protocol.

Three strategies - lexical, dense and their fusion - are interchangeable
because they all satisfy :class:`Retriever`. The evaluation harness in Phase 3
resolves a retriever by name and measures it through this interface, which is
the whole reason the interface exists.

**The failure this design prevents.** The usual way retrieval evaluation goes
wrong is not a bug in the metric, it is that the harness re-implements
retrieval. A script builds its own query, calls the search engine directly,
scores the result, and reports a number - and that number describes the script,
not the system. The two drift the moment production changes a boost, a filter
or an analyser, and nothing fails when they do. Making the API endpoint and the
harness both call the same object means a measured improvement is an
improvement to the code users hit.

**Why a Protocol rather than a base class.** Nothing here needs shared
behaviour, only a shared shape. A Protocol says "anything with this method
qualifies" without imposing an inheritance hierarchy, so a test double is a
five-line class rather than a subclass carrying machinery it does not want.
:data:`Retriever` is ``runtime_checkable`` so a wiring mistake fails an
``isinstance`` assertion in a test rather than an ``AttributeError`` in a view.

**Ranks are 1-based and contiguous.** This is a load-bearing convention, not a
formatting choice: Reciprocal Rank Fusion divides by ``k + rank``, so a 0-based
rank would give the top document a different weight than the algorithm intends,
and a gap in the sequence quietly changes the fused score of everything below
it. :func:`rerank` exists so no retriever has to remember this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# How deep each retriever goes before fusion, regardless of how many results
# the caller asked for.
#
# RRF can only reward a document that at least one retriever returned. If both
# lists are cut to 10 and the two retrievers disagree - which is precisely the
# case fusion exists to handle - a document ranked 12th by one and 1st by the
# other never enters the calculation at all, and hybrid search degenerates into
# an expensive way of running two searches. Fusing depth 50 and returning 10 is
# the standard shape for exactly this reason.
DEFAULT_FUSION_DEPTH = 50

# What a caller gets back unless they say otherwise.
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved paper.

    Keyed by ``arxiv_id`` rather than chunk ID because relevance is judged per
    paper: the golden set records ``relevant_ids`` as arXiv IDs, so a result
    identified any other way cannot be scored without a translation step that
    would itself need testing.

    ``score`` is comparable *within* one retriever's list and meaningless
    *across* retrievers - a BM25 score of 14.2 and a cosine similarity of 0.82
    say nothing about each other. That incomparability is the reason fusion
    uses :attr:`rank` and ignores :attr:`score` entirely. The score is carried
    anyway because it is what makes a surprising ranking diagnosable.
    """

    arxiv_id: str
    score: float
    rank: int
    debug: dict[str, Any] = field(default_factory=dict)

    def with_rank(self, rank: int) -> RetrievalResult:
        """A copy at a new rank. Frozen, so reranking rebuilds rather than mutates."""
        return RetrievalResult(
            arxiv_id=self.arxiv_id,
            score=self.score,
            rank=rank,
            debug=dict(self.debug),
        )


@runtime_checkable
class Retriever(Protocol):
    """Anything that can turn a query into a ranked list of papers."""

    name: str

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Return at most ``top_k`` results, best first, ranked from 1."""
        ...


def collapse_to_papers(
    candidates: list[tuple[str, float, dict[str, Any]]],
) -> list[RetrievalResult]:
    """Reduce ranked chunk hits to ranked papers, keeping each paper's best chunk.

    ``candidates`` is ``(arxiv_id, score, debug)`` already ordered best first.

    **Why this is necessary and why it is easy to miss.** Both retrievers search
    chunks. The schema allows a paper to have several, and 69 papers in the
    corpus genuinely do, because their abstracts exceeded the model's context
    window and were split. Without this step such a paper enters fusion once per
    chunk and collects an RRF contribution for each, so it outranks a
    single-chunk paper that matched better. The system would then be rewarding
    length rather than relevance - and the resulting metrics would look entirely
    reasonable, because every individual number in them would be correct.

    Nothing errors, no test fails, and the ablation table quietly measures the
    wrong thing. It is worth being explicit that this is the class of bug the
    whole project is arguing about.
    """
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    order: list[str] = []
    for arxiv_id, score, debug in candidates:
        if arxiv_id in best:
            # Already seen at a better rank, since the input is sorted. Record
            # that the paper matched more than once - useful when a result's
            # position looks wrong - but do not let it score twice.
            best[arxiv_id][1].setdefault("extra_chunk_hits", 0)
            best[arxiv_id][1]["extra_chunk_hits"] += 1
            continue
        best[arxiv_id] = (score, dict(debug))
        order.append(arxiv_id)

    return [
        RetrievalResult(
            arxiv_id=arxiv_id, score=best[arxiv_id][0], rank=rank, debug=best[arxiv_id][1]
        )
        for rank, arxiv_id in enumerate(order, start=1)
    ]


def rerank(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Renumber a list so ranks are 1-based and contiguous.

    Call this after any filtering or truncation. See the module docstring for
    why gaps in the sequence are not cosmetic.
    """
    return [result.with_rank(rank) for rank, result in enumerate(results, start=1)]


__all__ = [
    "DEFAULT_FUSION_DEPTH",
    "DEFAULT_TOP_K",
    "RetrievalResult",
    "Retriever",
    "collapse_to_papers",
    "rerank",
]
