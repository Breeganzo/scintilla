"""Rendering the answering evaluation.

Split from the harness for the same reason the retrieval report is: measuring
and formatting fail differently, and a change to a table should not be able to
break a number.
"""

from __future__ import annotations

from typing import Any

from evaluation.answer_harness import AnswerReport
from evaluation.report import _table


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def headline_table(report: AnswerReport) -> str:
    """The four numbers that decide whether the answering layer is safe."""
    rows = [
        [
            "abstained on unanswerable",
            f"{sum(o.abstained for o in report.unanswerable)}/{len(report.unanswerable)}",
            _pct(report.abstention_rate),
            "higher is better",
        ],
        [
            "abstained on answerable",
            f"{sum(o.abstained for o in report.answerable)}/{len(report.answerable)}",
            _pct(report.false_abstention_rate),
            "lower is better",
        ],
        [
            "citation precision",
            f"{len(report.answered) - report.uncited_answers} answers cited",
            _pct(report.citation_precision),
            "cited a passage that existed",
        ],
        [
            "citation grounding",
            f"{len(report.answerable)} answerable",
            _pct(report.citation_grounding),
            "cited a paper labelled relevant",
        ],
        [
            "answers with no citation",
            str(report.uncited_answers),
            "-",
            "least grounded output possible",
        ],
        [
            "hallucinated citation numbers",
            str(report.hallucinated_citations),
            "-",
            "pointed at a passage never supplied",
        ],
    ]
    return _table(["measure", "count", "rate", "reading"], rows)


def failure_table(report: AnswerReport) -> str:
    """Every query where the abstention decision went the wrong way.

    Listed individually rather than summarised because ten unanswerable queries
    is a small enough sample that a rate hides more than it shows: knowing
    *which* question was answered anyway is what makes the failure fixable.
    """
    rows = [
        [
            o.query_id,
            o.query_class,
            o.query[:52],
            "answered anyway" if o.answerable is False else "refused a real question",
        ]
        for o in report.outcomes
        if not o.correct
    ]
    if not rows:
        return "_No abstention failures: every unanswerable query was refused and every answerable one attempted._"
    return _table(["id", "class", "query", "what went wrong"], rows)


def to_dict(report: AnswerReport) -> dict[str, Any]:
    """A JSON-safe record, detailed enough to diff two runs."""
    return {
        "mode": report.mode,
        "model": report.model,
        "top_k": report.top_k,
        "context_size": report.context_size,
        "golden_set_version": report.golden_set_version,
        "degraded": report.degraded,
        "usage": report.usage,
        "summary": {
            "abstention_rate": report.abstention_rate,
            "false_abstention_rate": report.false_abstention_rate,
            "citation_precision": report.citation_precision,
            "citation_grounding": report.citation_grounding,
            "uncited_answers": report.uncited_answers,
            "hallucinated_citations": report.hallucinated_citations,
            "median_latency_ms": report.median_latency_ms,
            "answerable": len(report.answerable),
            "unanswerable": len(report.unanswerable),
        },
        "queries": [
            {
                "query_id": o.query_id,
                "query_class": o.query_class,
                "answerable": o.answerable,
                "abstained": o.abstained,
                "correct": o.correct,
                "degraded": o.result.degraded,
                "citations": o.result.citations,
                "invalid_citations": o.result.invalid_citations,
                "cited_ids": o.result.cited_ids,
                "cited_relevant": o.cited_relevant,
                "elapsed_ms": o.result.elapsed_ms,
                "answer": o.result.text,
            }
            for o in report.outcomes
        ],
    }


def to_markdown(report: AnswerReport) -> str:
    parts = [
        "# Answering evaluation",
        "",
        f"Retriever `{report.mode}` at top-{report.top_k}, "
        f"{report.context_size} passages in the prompt, model `{report.model}`, "
        f"golden set v{report.golden_set_version}.",
        "",
        "## Headline",
        "",
        headline_table(report),
        "",
        "## Abstention failures",
        "",
        failure_table(report),
        "",
        "## What is not measured here",
        "",
        "Citation *precision* asks whether a cited number referred to a passage that",
        "was actually supplied, and citation *grounding* asks whether that passage was",
        "a paper the golden set labels relevant. Neither asks whether the cited passage",
        "genuinely supports the sentence it is attached to. That requires a human",
        "reader or a judge model, and neither has been used, so no claim is made about",
        "it.",
    ]
    return "\n".join(parts) + "\n"


__all__ = ["failure_table", "headline_table", "to_dict", "to_markdown"]
