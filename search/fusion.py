"""Reciprocal Rank Fusion.

**The algorithm.** For each document, sum ``1 / (k + rank)`` over every ranked
list it appears in, where ``rank`` is 1-based. Sort by that sum, descending.
That is the whole thing.

**Why fuse ranks instead of scores.** The obvious approach is to normalise both
score distributions and take a weighted average. It does not work. BM25 scores
are unbounded and depend on corpus statistics, so the same document scores
differently as the corpus grows - and this corpus grows every morning at 06:00.
Cosine similarities sit in a fixed range with a completely different shape.
Min-max normalising either one makes the top score depend on the *worst* result
retrieved, which means adding an irrelevant document at rank 50 changes the
score at rank 1. Every fix for that requires assumptions about the distributions
that stop holding when the corpus or the model changes.

RRF sidesteps all of it by throwing the scores away. Rank 1 from BM25 and rank 1
from dense retrieval contribute exactly the same amount, whatever the underlying
numbers were. That also means RRF cannot tell a confident match from a marginal
one, which is a real cost and worth stating rather than glossing: it is the
price of not having to make the distributions comparable.

**Why k = 60.** It comes from Cormack, Clarke and Buettcher (SIGIR 2009), who
found it worked well across TREC runs and noted the method was insensitive to
the exact value. What ``k`` actually controls is how sharply the top of each
list dominates. At ``k = 0`` rank 1 contributes 1.0 and rank 2 contributes 0.5,
so a single retriever being confidently wrong drags the fused list with it. At
``k = 60`` those become 0.0164 and 0.0161 - nearly equal, so the fused ordering
is driven by *agreement across lists* rather than by either list's top hit.

Concretely: with ``k = 60``, a document ranked 1st by one retriever and absent
from the other scores 0.0164, while a document ranked 3rd by both scores 0.0317
and wins. That is the intended behaviour - two independent signals agreeing is
better evidence than one signal shouting.

It is exposed as a parameter because it is exactly the kind of constant that
should be measurable rather than asserted, and Day 9 has the harness to measure
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from search.retrievers.base import RetrievalResult

# See the module docstring. Not a tuning knob that was swept - the conventional
# default, adopted deliberately and left measurable.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    runs: dict[str, Sequence[RetrievalResult]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse named ranked lists into one.

    ``runs`` maps a retriever name to its results. The name is kept because it
    ends up in ``debug`` as the per-retriever rank breakdown, which is what
    makes a surprising fused position explainable rather than mysterious - and
    it is the data the "why this result" panel renders on Day 11.

    Ties are broken by arXiv ID. Floating-point sums make exact ties rare but
    not impossible, and an unstable order would make the same query return
    different rankings on different runs, which would show up in evaluation as
    inexplicable metric drift.
    """
    if k < 0:
        raise ValueError(f"RRF k must be non-negative, got {k}")

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, Any]] = {}

    for run_name, results in runs.items():
        for result in results:
            if result.rank < 1:
                raise ValueError(
                    f"RRF needs 1-based ranks; {run_name} returned rank "
                    f"{result.rank} for {result.arxiv_id}. A 0-based rank would "
                    f"silently overweight the top hit."
                )
            contribution = 1.0 / (k + result.rank)
            scores[result.arxiv_id] = scores.get(result.arxiv_id, 0.0) + contribution
            detail = contributions.setdefault(result.arxiv_id, {})
            detail[run_name] = {
                "rank": result.rank,
                "score": result.score,
                "contribution": contribution,
            }

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        RetrievalResult(
            arxiv_id=arxiv_id,
            score=score,
            rank=rank,
            debug={
                "rrf_k": k,
                # Which retrievers found this and where. A document present in
                # one list only is the interesting case, and this is what makes
                # it visible at a glance.
                "runs": contributions[arxiv_id],
            },
        )
        for rank, (arxiv_id, score) in enumerate(ordered, start=1)
    ]


__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]
