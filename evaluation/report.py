"""Turning reports into JSON and markdown.

Two outputs with different jobs. The JSON is the record: complete, per query,
diffable, and what Day 10's regression check reads. The markdown is the
argument: small enough to paste into a README and read in one pass.

Keeping them separate matters because the temptation is to publish only the
summary. A table showing hybrid ahead is not evidence unless the per-query
detail behind it can be inspected - which is the same objection this project
makes to systems that report no numbers at all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from evaluation.harness import ModeReport
from evaluation.significance import compare

# The columns worth arguing about. Every metric is computed and stored; only
# these are printed, because a table with eleven columns does not get read.
HEADLINE = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")

# Ordered so a reader meets the baselines before the thing being advocated for.
CLASS_ORDER = ("exact_term", "paraphrase", "conceptual", "multi_hop")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A markdown table, right-aligned except the first column."""
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _best(reports: Mapping[str, ModeReport], metric: str) -> str:
    """Which mode leads on a metric. Ties are reported as ties, not broken."""
    scores = {mode: report.overall.get(metric, 0.0) for mode, report in reports.items()}
    top = max(scores.values())
    leaders = [mode for mode, score in scores.items() if score == top]
    return ", ".join(leaders)


def headline_table(reports: Mapping[str, ModeReport]) -> str:
    """One row per mode over the 40 answerable queries."""
    rows = []
    for mode, report in reports.items():
        overall = report.overall
        rows.append(
            [
                mode,
                *[_fmt(overall.get(metric, 0.0)) for metric in HEADLINE],
                f"{report.median_latency_ms:.0f}",
            ]
        )
    return _table(["mode", *HEADLINE, "median ms"], rows)


def per_class_table(reports: Mapping[str, ModeReport], metric: str = "recall@10") -> str:
    """One row per query class, one column per mode.

    This is the table that decides whether hybrid search earned its complexity.
    Fusion is supposed to help where lexical and dense disagree - paraphrase and
    conceptual queries - and to cost something where exact term matching already
    works. If those two effects are not visible here, the headline average is
    hiding them.
    """
    modes = list(reports)
    by_class = {mode: report.by_class() for mode, report in reports.items()}

    present = [name for name in CLASS_ORDER if any(name in by_class[mode] for mode in modes)]
    extra = sorted({name for mode in modes for name in by_class[mode]} - set(CLASS_ORDER))

    rows = []
    for name in [*present, *extra]:
        cells = [_fmt(by_class[mode].get(name, {}).get(metric, 0.0)) for mode in modes]
        rows.append([name, *cells])
    return _table(["class", *modes], rows)


def where_hybrid_loses(
    reports: Mapping[str, ModeReport], metric: str = "ndcg@10"
) -> list[dict[str, Any]]:
    """Queries where a baseline beat hybrid, worst deficit first.

    The point of naming them is that fusion has a known failure mode: a document
    both retrievers rank mediocrely can be overtaken by one that a single
    retriever ranked first, because RRF rewards agreement. Without the per-query
    list this shows up only as a slightly lower average, which is easy to
    explain away.
    """
    if "hybrid" not in reports:
        return []

    hybrid = {outcome.query_id: outcome for outcome in reports["hybrid"].scored}
    baselines = {
        mode: {o.query_id: o for o in report.scored}
        for mode, report in reports.items()
        if mode != "hybrid"
    }

    losses = []
    for query_id, outcome in hybrid.items():
        mine = outcome.metrics.get(metric, 0.0)
        rivals = {
            mode: rows[query_id].metrics.get(metric, 0.0)
            for mode, rows in baselines.items()
            if query_id in rows
        }
        if not rivals:
            continue
        best_mode = max(rivals, key=lambda mode: rivals[mode])
        if rivals[best_mode] > mine:
            losses.append(
                {
                    "query_id": query_id,
                    "query_class": outcome.query_class,
                    "query": outcome.query,
                    "metric": metric,
                    "hybrid": mine,
                    "beaten_by": best_mode,
                    "their_score": rivals[best_mode],
                    "deficit": rivals[best_mode] - mine,
                    "hybrid_first_hit_rank": outcome.first_hit_rank,
                    "missed": outcome.misses,
                }
            )
    return sorted(losses, key=lambda row: row["deficit"], reverse=True)


def aligned_scores(report: ModeReport, metric: str) -> tuple[list[str], list[float]]:
    """Per-query scores in a fixed query order, so two modes can be paired.

    Sorted by query id rather than trusting insertion order. Comparing two
    unaligned lists would silently pair query 1 of one mode with query 7 of
    another and produce a confidence interval for nothing at all.
    """
    rows = sorted(report.scored, key=lambda outcome: outcome.query_id)
    return [row.query_id for row in rows], [row.metrics.get(metric, 0.0) for row in rows]


def significance_table(reports: Mapping[str, ModeReport], metric: str = "ndcg@10") -> str:
    """Every pair of modes, with a paired bootstrap interval on the difference.

    Without this the table invites a reader to treat 0.691 against 0.649 as a
    settled result. With forty queries it may or may not be; the interval says
    which, and a difference whose interval spans zero is reported as undecided
    rather than quietly rounded into a claim.
    """
    rows = []
    for left, right in combinations(reports, 2):
        ids_left, scores_left = aligned_scores(reports[left], metric)
        ids_right, scores_right = aligned_scores(reports[right], metric)
        if ids_left != ids_right:
            raise ValueError(f"{left} and {right} were scored on different queries")

        result = compare(scores_left, scores_right)
        rows.append(
            [
                f"{right} - {left}",
                f"{result.difference:+.3f}",
                f"[{result.ci_low:+.3f}, {result.ci_high:+.3f}]",
                f"{result.p_value:.4f}",
                f"{result.wins_b}/{result.wins_a}/{result.ties}",
                "yes" if result.significant else "no",
            ]
        )
    return _table(
        [f"comparison ({metric})", "difference", "95% CI", "p", "W/L/T", "significant"],
        rows,
    )


def unanswerable_summary(reports: Mapping[str, ModeReport]) -> str:
    """What the modes do when there is nothing to find.

    Every retriever returns its top ``k`` regardless, so the honest statement is
    that retrieval alone cannot pass this class - the number below is the count
    of confidently wrong results a user would see with no answering layer in
    front of them. It is the case for Day 10's abstention, stated as a number.
    """
    rows = []
    for mode, report in reports.items():
        unanswerable = report.unanswerable
        returned = sum(len(outcome.retrieved_ids) for outcome in unanswerable)
        empty = sum(1 for outcome in unanswerable if not outcome.retrieved_ids)
        rows.append([mode, str(len(unanswerable)), str(returned), str(empty)])
    return _table(["mode", "queries", "results returned", "returned nothing"], rows)


def to_json(reports: Mapping[str, ModeReport], meta: Mapping[str, Any] | None = None) -> str:
    """The complete record: every query, every score, every retrieved id."""
    payload: dict[str, Any] = {"meta": dict(meta or {}), "modes": {}}
    for mode, report in reports.items():
        payload["modes"][mode] = {
            "mode": report.mode,
            "top_k": report.top_k,
            "golden_set_version": report.golden_set_version,
            "overall": report.overall,
            "by_class": report.by_class(),
            "median_latency_ms": report.median_latency_ms,
            "queries": [
                {
                    "query_id": outcome.query_id,
                    "query_class": outcome.query_class,
                    "query": outcome.query,
                    "retrieved_ids": outcome.retrieved_ids,
                    "relevant_ids": outcome.relevant_ids,
                    "metrics": outcome.metrics,
                    "first_hit_rank": outcome.first_hit_rank,
                    "misses": outcome.misses,
                    "elapsed_ms": round(outcome.elapsed_ms, 2),
                }
                for outcome in report.outcomes
            ],
        }
    return json.dumps(payload, indent=2, sort_keys=False)


def to_markdown(reports: Mapping[str, ModeReport], meta: Mapping[str, Any] | None = None) -> str:
    """The readable summary, including the parts that are unflattering."""
    meta = dict(meta or {})
    first = next(iter(reports.values()))
    scored = len(first.scored)
    unanswerable = len(first.unanswerable)

    parts = [
        "## Retrieval ablation",
        "",
        f"Golden set v{first.golden_set_version}: {scored} answerable queries scored, "
        f"{unanswerable} unanswerable queries held out of the averages "
        f"(recall over an empty relevant set is 0/0, not 0).",
        "",
        f"Corpus: {meta.get('corpus_papers', '?')} papers / {meta.get('corpus_chunks', '?')} chunks. "
        f"top_k={first.top_k}. Commit `{meta.get('git_sha', 'unknown')[:8]}`.",
        "",
        headline_table(reports),
        "",
        "### recall@10 by query class",
        "",
        per_class_table(reports, "recall@10"),
        "",
        "### nDCG@10 by query class",
        "",
        per_class_table(reports, "ndcg@10"),
        "",
        "### Is the difference real?",
        "",
        "Paired bootstrap over the same 40 queries, 10,000 resamples, fixed seed. "
        "W/L/T counts queries where the right-hand mode won, lost or tied.",
        "",
        significance_table(reports, "ndcg@10"),
        "",
        "### Unanswerable queries",
        "",
        unanswerable_summary(reports),
        "",
    ]

    losses = where_hybrid_loses(reports)
    parts += ["### Where hybrid loses", ""]
    if not losses:
        parts += ["Hybrid was at least as good as both baselines on every scored query.", ""]
    else:
        rows = [
            [
                row["query_id"],
                row["query_class"],
                _fmt(row["hybrid"]),
                row["beaten_by"],
                _fmt(row["their_score"]),
                _fmt(row["deficit"]),
            ]
            for row in losses
        ]
        parts += [
            f"{len(losses)} of {scored} scored queries, by nDCG@10.",
            "",
            _table(["query", "class", "hybrid", "beaten by", "their score", "deficit"], rows),
            "",
        ]

    parts += [
        "Leaders: " + "; ".join(f"{metric} {_best(reports, metric)}" for metric in HEADLINE) + ".",
        "",
    ]
    return "\n".join(parts)


__all__ = [
    "CLASS_ORDER",
    "HEADLINE",
    "aligned_scores",
    "headline_table",
    "per_class_table",
    "significance_table",
    "to_json",
    "to_markdown",
    "unanswerable_summary",
    "where_hybrid_loses",
]
