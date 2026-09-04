"""Retrieval metrics, implemented from scratch.

**Why not import a library.** ``pytrec_eval`` and ``ranx`` are correct and
well tested, and using one would be the right call in most production settings.
They are avoided here for a specific reason: the argument this project makes is
that most RAG systems ship without knowing whether retrieval works. Being able
to say "recall@10 is 0.71" while treating the metric as a black box is a weaker
version of the same problem. Every number in the ablation table can be derived
by hand from the definitions below, and the tests do exactly that.

**What each metric answers, in one line.**

- ``recall@k`` - of the papers that *should* have come back, how many did?
- ``precision@k`` - of the papers that *did* come back, how many should have?
- ``MRR`` - how far down the list is the first useful result?
- ``nDCG@k`` - the same as recall, but paying more for a good result at rank 1
  than the same result at rank 10.

**Why all four rather than one.** They disagree, and the disagreements are the
findings. A retriever that puts one relevant paper at rank 1 and misses four
others scores well on MRR and badly on recall. A retriever that finds all five
but buries them at ranks 6-10 does the reverse. Reporting a single number lets
you pick the one that flatters the system, which is how "we improved retrieval"
gets claimed without evidence.

**Binary relevance.** The golden set records relevant / not relevant with no
grades, so every gain is 0 or 1. nDCG supports graded relevance and would be
more informative with it; grading 163 labels consistently by one judge was not
credible, and the bias documentation says so. Binary gains keep the metric
honest about the labels underneath it.

**Rank convention.** Positions are 1-based, matching
:class:`~search.retrievers.base.RetrievalResult`. Using 0-based positions here
would make ``1 / (rank + 1)`` and ``log2(rank + 1)`` silently mean different
things than the literature does, and the resulting numbers would be plausible,
wrong, and impossible to compare to any published baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# Cut-offs reported for every run. 1 is "did it nail it", 5 is roughly what a
# user reads, 10 is what the API returns by default and what the answering
# layer will be given as context on Day 10.
DEFAULT_KS = (1, 5, 10)


class MetricError(ValueError):
    """Raised when a metric is asked for something it cannot mean.

    A metric that returns a number for nonsense input is worse than one that
    refuses, because the number reaches the table and nobody can tell it apart
    from a real measurement.
    """


def _check(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> frozenset[str]:
    """Shared preconditions. Returns the relevant set.

    Three things are rejected rather than tolerated:

    ``k < 1``
        There is no such thing as the top zero results.

    Duplicates in ``retrieved``
        A paper appearing twice would be counted twice by ``recall@k``, pushing
        it above 1.0 - or, worse, sitting just under 1.0 and looking fine.
        :func:`~search.retrievers.base.collapse_to_papers` exists to prevent
        this upstream, so a duplicate here means that collapse was skipped and
        the whole run is measuring the wrong thing.

    Empty ``relevant``
        Recall is a fraction of the relevant set, and that fraction is 0/0.
        Returning 0.0 would punish a retriever for a query with no answer;
        returning 1.0 would reward it for anything at all. Both are defensible
        in isolation, which is exactly why the choice must not be buried in a
        helper - the ``unanswerable`` class is measured by whether the system
        abstains, not by what it ranked.
    """
    if k < 1:
        raise MetricError(f"k must be at least 1, got {k}")

    if len(set(retrieved)) != len(retrieved):
        duplicates = sorted({item for item in retrieved if retrieved.count(item) > 1})
        raise MetricError(
            f"retrieved contains duplicates: {duplicates}. Results must be collapsed "
            "to one entry per paper before scoring, or recall is inflated."
        )

    relevant_set = frozenset(relevant)
    if not relevant_set:
        raise MetricError(
            "relevant is empty, so recall and nDCG are 0/0. Unanswerable queries "
            "are scored by abstention, not by ranking - handle them separately."
        )

    return relevant_set


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant papers that appear in the top ``k``.

    Capped by ``k``: a query with 8 relevant papers cannot exceed 0.625 at
    ``k=5`` no matter how good the retriever is. That ceiling is a property of
    the golden set, not of the system, which is why per-class recall is only
    meaningful against other modes on the *same* queries.
    """
    relevant_set = _check(retrieved, relevant, k)
    found = sum(1 for arxiv_id in retrieved[:k] if arxiv_id in relevant_set)
    return found / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    Divided by ``k`` rather than by the number of results actually returned.
    Dividing by the latter would let a retriever that returns two results, both
    relevant, score 1.0 against one that returns ten with the same two at the
    top - rewarding timidity rather than accuracy.
    """
    relevant_set = _check(retrieved, relevant, k)
    found = sum(1 for arxiv_id in retrieved[:k] if arxiv_id in relevant_set)
    return found / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """``1 / rank`` of the first relevant result, or 0.0 if none is in the top ``k``.

    Deliberately harsh and deliberately shallow: 1.0 for a hit at rank 1, 0.5 at
    rank 2, 0.1 at rank 10. It models a reader who scans from the top and stops
    at the first thing that helps, and it ignores everything after that hit -
    which is why it must not be reported alone.
    """
    relevant_set = _check(retrieved, relevant, k)
    for rank, arxiv_id in enumerate(retrieved[:k], start=1):
        if arxiv_id in relevant_set:
            return 1 / rank
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain over a list of gains in rank order.

    The discount is ``1 / log2(rank + 1)``, so rank 1 is undiscounted (log2(2)
    = 1), rank 2 keeps ~63% of its gain and rank 10 keeps ~29%. Taken as given
    from the literature rather than derived - the point of the log is that the
    penalty for slipping from rank 1 to 2 should be larger than from 9 to 10,
    and any smoothly decaying function would express that.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG@k with binary gains.

    Normalised against the best ordering the golden set permits: all relevant
    papers first, so the ideal list has ``min(len(relevant), k)`` ones. Without
    that normalisation a query with 8 relevant papers would always outscore one
    with 2, and the per-class averages would mostly reflect how generously each
    class happened to be labelled.
    """
    relevant_set = _check(retrieved, relevant, k)

    gains = [1.0 if arxiv_id in relevant_set else 0.0 for arxiv_id in retrieved[:k]]
    ideal = [1.0] * min(len(relevant_set), k)

    ideal_dcg = dcg(ideal)
    if not ideal_dcg:  # pragma: no cover - unreachable, _check rejects empty relevant
        return 0.0
    return dcg(gains) / ideal_dcg


def score_query(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Every metric for one query, keyed for the results table.

    MRR is measured at the largest cut-off. Computing it at each ``k`` as well
    would add columns that are almost always equal - the first relevant result
    is usually inside the top 1 or nowhere - and a table nobody reads is not
    more rigorous than one they do.
    """
    deepest = max(ks)
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    scores["mrr"] = reciprocal_rank(retrieved, relevant, deepest)
    return scores


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence.

    Written out rather than imported from ``statistics`` because
    ``statistics.mean([])`` raises, and an empty class in a filtered report is a
    normal thing to render, not an error.
    """
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate(per_query: Iterable[dict[str, float]]) -> dict[str, float]:
    """Macro-average the per-query scores.

    Macro rather than micro: every query counts once, regardless of how many
    relevant papers it has. Micro-averaging would let the handful of queries
    with a dozen labels dominate, so the headline number would describe the
    labelling effort rather than the retriever.
    """
    rows = list(per_query)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: mean(row[key] for row in rows) for key in keys}


__all__ = [
    "DEFAULT_KS",
    "MetricError",
    "aggregate",
    "dcg",
    "mean",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_query",
]
