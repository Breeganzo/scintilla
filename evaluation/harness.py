"""Running a retrieval mode over the golden set.

**The one design rule.** The harness resolves a retriever through
:func:`search.retrievers.get_retriever` - the same call the API view makes. It
does not build its own query, hit OpenSearch directly, or reimplement fusion.
The usual way an ablation becomes fiction is that the harness and the product
drift apart: someone tunes a boost in the view, the script keeps its own copy,
and the table goes on describing code nobody runs. Sharing the entry point makes
that impossible rather than merely discouraged.

**Unanswerable queries take a different road.** Ten of the fifty have no
relevant papers, and recall over an empty relevant set is 0/0. They are not
scored with ranking metrics and not folded into the headline average; they are
carried separately, recording what the retriever returned and how confident it
was, which is the input Day 10's abstention measurement needs. Averaging them in
as zeros would drag every mode down by the same amount and tell you nothing,
while averaging them in as ones would hand every mode a fifth of a free point.

**Warm-up is not a nicety.** The first dense query loads a ~130 MB sentence
transformer, which took 12.2 seconds against this corpus while every subsequent
query took ~65 ms. Timed cold, the first query in the sweep would carry the
model load and ``dense`` would look two orders of magnitude slower than it is.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from evaluation.golden import GoldenQuery, GoldenSet, load_golden_set
from evaluation.metrics import DEFAULT_KS, aggregate, score_query
from search.retrievers import DEFAULT_TOP_K, MODES, Retriever, get_retriever

# Sent before the sweep starts so the first measured query is not paying for a
# model load. The text is deliberately unlike anything in the golden set so a
# warm cache cannot flatter a real query.
WARMUP_QUERY = "warm up the embedding model before anything is timed"


@dataclass(frozen=True)
class QueryOutcome:
    """What one query did in one run."""

    query_id: str
    query_class: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: dict[str, float]
    elapsed_ms: float

    @property
    def is_unanswerable(self) -> bool:
        return not self.relevant_ids

    @property
    def hits(self) -> list[str]:
        """Retrieved papers that were labelled relevant, in rank order."""
        relevant = set(self.relevant_ids)
        return [arxiv_id for arxiv_id in self.retrieved_ids if arxiv_id in relevant]

    @property
    def misses(self) -> list[str]:
        """Labelled papers the retriever never returned."""
        retrieved = set(self.retrieved_ids)
        return [arxiv_id for arxiv_id in self.relevant_ids if arxiv_id not in retrieved]

    @property
    def first_hit_rank(self) -> int | None:
        """1-based position of the first relevant result, or None."""
        relevant = set(self.relevant_ids)
        for rank, arxiv_id in enumerate(self.retrieved_ids, start=1):
            if arxiv_id in relevant:
                return rank
        return None


@dataclass(frozen=True)
class ModeReport:
    """Every outcome for one mode, plus the aggregates derived from them."""

    mode: str
    top_k: int
    golden_set_version: int
    outcomes: list[QueryOutcome] = field(default_factory=list)

    @property
    def scored(self) -> list[QueryOutcome]:
        """The 40 answerable queries - the only ones ranking metrics apply to."""
        return [outcome for outcome in self.outcomes if not outcome.is_unanswerable]

    @property
    def unanswerable(self) -> list[QueryOutcome]:
        return [outcome for outcome in self.outcomes if outcome.is_unanswerable]

    @property
    def overall(self) -> dict[str, float]:
        return aggregate(outcome.metrics for outcome in self.scored)

    def by_class(self) -> dict[str, dict[str, float]]:
        """Aggregates per query class, in the golden set's declared order.

        The per-class split is where the argument for hybrid search either
        survives or does not. A single headline number can hide a mode winning
        on the class it was always going to win and losing everywhere else.
        """
        classes: dict[str, list[QueryOutcome]] = {}
        for outcome in self.scored:
            classes.setdefault(outcome.query_class, []).append(outcome)
        return {
            name: aggregate(outcome.metrics for outcome in group) for name, group in classes.items()
        }

    @property
    def median_latency_ms(self) -> float:
        """Median, not mean: one slow query should not redefine the mode.

        Latency here is retrieval only and measured in-process, so it excludes
        HTTP, serialisation and hydration. It is useful for comparing modes to
        each other and is not a claim about what a user experiences.
        """
        timings = sorted(outcome.elapsed_ms for outcome in self.outcomes)
        if not timings:
            return 0.0
        middle = len(timings) // 2
        if len(timings) % 2:
            return timings[middle]
        return (timings[middle - 1] + timings[middle]) / 2


def run_query(
    retriever: Retriever,
    query: GoldenQuery,
    top_k: int,
    ks: Sequence[int],
) -> QueryOutcome:
    """Retrieve for one golden query and score it."""
    started = time.perf_counter()
    results = retriever.retrieve(query.query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000

    retrieved_ids = [result.arxiv_id for result in results]

    # Unanswerable queries carry no ranking metrics at all. An empty dict is
    # more honest than zeros, which would be indistinguishable from a mode that
    # tried and failed.
    metrics: dict[str, float] = {}
    if query.relevant_ids:
        metrics = score_query(retrieved_ids, query.relevant_ids, ks=ks)

    return QueryOutcome(
        query_id=query.id,
        query_class=query.query_class,
        query=query.query,
        retrieved_ids=retrieved_ids,
        relevant_ids=list(query.relevant_ids),
        metrics=metrics,
        elapsed_ms=elapsed_ms,
    )


def run_mode(
    mode: str,
    golden: GoldenSet | None = None,
    top_k: int = DEFAULT_TOP_K,
    ks: Sequence[int] = DEFAULT_KS,
    warm_up: bool = True,
    retriever: Retriever | None = None,
    label: str | None = None,
) -> ModeReport:
    """Sweep one retrieval mode over the whole golden set.

    ``retriever`` may be supplied directly to score a configuration the registry
    does not name - sweeping RRF's ``k``, for instance. ``label`` then carries
    the variant's name into the report. This is the only sanctioned way to
    measure something the API cannot serve, and the label exists so such a row
    can never be mistaken for a shipped mode in a results table.
    """
    golden = golden or load_golden_set()
    retriever = retriever or get_retriever(mode)

    if warm_up:
        retriever.retrieve(WARMUP_QUERY, top_k=1)

    outcomes = [run_query(retriever, query, top_k, ks) for query in golden]
    return ModeReport(
        mode=label or mode,
        top_k=top_k,
        golden_set_version=golden.version,
        outcomes=outcomes,
    )


def run_ablation(
    modes: Iterable[str] = MODES,
    golden: GoldenSet | None = None,
    top_k: int = DEFAULT_TOP_K,
    ks: Sequence[int] = DEFAULT_KS,
    warm_up: bool = True,
) -> dict[str, ModeReport]:
    """Every mode over the same golden set, same cut-offs, same process.

    Loading the golden set once rather than per mode is not an optimisation: it
    guarantees all three modes were scored against byte-identical labels, so a
    difference in the table is a difference in retrieval.
    """
    golden = golden or load_golden_set()
    return {
        mode: run_mode(mode, golden=golden, top_k=top_k, ks=ks, warm_up=warm_up) for mode in modes
    }


__all__ = [
    "WARMUP_QUERY",
    "ModeReport",
    "QueryOutcome",
    "run_ablation",
    "run_mode",
    "run_query",
]
