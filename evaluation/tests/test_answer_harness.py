"""Tests for the answering evaluation harness.

The rates being computed here are simple arithmetic, which is exactly why they
are tested: an abstention rate that divides by the wrong denominator produces a
number that looks entirely reasonable and is wrong, and there is nothing in a
report to reveal it. The fixtures below are constructed so every rate has a
value that can be checked by counting on one hand.
"""

from __future__ import annotations

import pytest

from evaluation.answer_harness import AnswerOutcome, AnswerReport
from evaluation.answer_report import failure_table, headline_table, to_dict, to_markdown
from search.answering import AnswerResult, Passage


def outcome(
    query_id: str,
    answerable: bool,
    abstained: bool = False,
    citations: list[int] | None = None,
    invalid: list[int] | None = None,
    cited_arxiv: list[str] | None = None,
    relevant: list[str] | None = None,
    degraded: bool = False,
) -> AnswerOutcome:
    cited_arxiv = cited_arxiv if cited_arxiv is not None else []
    passages = [Passage(n + 1, arxiv, f"T{n}", "body") for n, arxiv in enumerate(cited_arxiv)]
    return AnswerOutcome(
        query_id=query_id,
        query_class="exact_term" if answerable else "unanswerable",
        query=f"query {query_id}",
        answerable=answerable,
        result=AnswerResult(
            query=f"query {query_id}",
            text="" if abstained else "An answer.",
            abstained=abstained,
            passages=passages,
            citations=citations or [],
            invalid_citations=invalid or [],
            degraded=degraded,
            model="stub-1",
            elapsed_ms=100,
        ),
        retrieved_ids=cited_arxiv,
        relevant_ids=relevant or [],
    )


def report(outcomes: list[AnswerOutcome]) -> AnswerReport:
    return AnswerReport(
        mode="dense",
        model="stub-1",
        top_k=10,
        context_size=5,
        golden_set_version=1,
        outcomes=outcomes,
    )


class TestAbstentionRates:
    def test_a_perfect_system_refuses_everything_unanswerable_and_nothing_else(self):
        r = report(
            [
                outcome("u1", answerable=False, abstained=True),
                outcome("u2", answerable=False, abstained=True),
                outcome("a1", answerable=True, citations=[1], cited_arxiv=["x"]),
                outcome("a2", answerable=True, citations=[1], cited_arxiv=["x"]),
            ]
        )
        assert r.abstention_rate == 1.0
        assert r.false_abstention_rate == 0.0
        assert all(o.correct for o in r.outcomes)

    def test_a_system_that_refuses_everything_scores_perfectly_on_one_and_worst_on_the_other(self):
        # The whole reason both numbers are reported. Either alone can be
        # gamed by a degenerate system; the pair cannot.
        r = report(
            [
                outcome("u1", answerable=False, abstained=True),
                outcome("a1", answerable=True, abstained=True),
                outcome("a2", answerable=True, abstained=True),
            ]
        )
        assert r.abstention_rate == 1.0
        assert r.false_abstention_rate == 1.0

    def test_a_system_that_never_refuses_scores_zero_on_abstention(self):
        r = report(
            [
                outcome("u1", answerable=False),
                outcome("u2", answerable=False),
                outcome("a1", answerable=True, citations=[1], cited_arxiv=["x"]),
            ]
        )
        assert r.abstention_rate == 0.0
        assert r.false_abstention_rate == 0.0

    def test_the_denominators_do_not_bleed_into_each_other(self):
        r = report(
            [
                outcome("u1", answerable=False, abstained=True),
                outcome("u2", answerable=False),
                outcome("a1", answerable=True),
                outcome("a2", answerable=True),
                outcome("a3", answerable=True, abstained=True),
            ]
        )
        assert r.abstention_rate == pytest.approx(0.5)
        assert r.false_abstention_rate == pytest.approx(1 / 3)


class TestCitationMeasures:
    def test_precision_averages_only_over_answers_that_cited(self):
        r = report(
            [
                outcome("a1", answerable=True, citations=[1, 2], cited_arxiv=["x", "y"]),
                outcome("a2", answerable=True, citations=[1, 9], invalid=[9], cited_arxiv=["x"]),
                outcome("a3", answerable=True),  # answered, cited nothing
            ]
        )
        # (1.0 + 0.5) / 2 - the uncited answer has no precision to average.
        assert r.citation_precision == pytest.approx(0.75)

    def test_uncited_answers_are_counted_rather_than_dropped(self):
        r = report(
            [
                outcome("a1", answerable=True, citations=[1], cited_arxiv=["x"]),
                outcome("a2", answerable=True),
                outcome("a3", answerable=True),
            ]
        )
        assert r.uncited_answers == 2

    def test_degraded_answers_are_excluded_from_the_answered_pool(self):
        # A request that never reached the model is not an uncited answer.
        r = report(
            [
                outcome("a1", answerable=True, degraded=True),
                outcome("a2", answerable=True, citations=[1], cited_arxiv=["x"]),
            ]
        )
        assert len(r.answered) == 1
        assert r.uncited_answers == 0

    def test_hallucinated_numbers_are_totalled_across_queries(self):
        r = report(
            [
                outcome("a1", answerable=True, citations=[9], invalid=[9], cited_arxiv=["x"]),
                outcome("a2", answerable=True, citations=[8, 8], invalid=[8, 8], cited_arxiv=["x"]),
            ]
        )
        assert r.hallucinated_citations == 3

    def test_grounding_asks_whether_the_cited_paper_was_actually_relevant(self):
        o = outcome(
            "a1",
            answerable=True,
            citations=[1, 2],
            cited_arxiv=["good", "bad"],
            relevant=["good"],
        )
        assert o.cited_relevant == 1
        assert o.citation_grounding == pytest.approx(0.5)

    def test_grounding_is_undefined_rather_than_zero_when_nothing_was_cited(self):
        o = outcome("a1", answerable=True, abstained=True, relevant=["good"])
        assert o.citation_grounding is None


class TestCorrectness:
    def test_answering_an_unanswerable_query_is_incorrect(self):
        assert outcome("u1", answerable=False, abstained=False).correct is False

    def test_refusing_an_answerable_query_is_incorrect(self):
        assert outcome("a1", answerable=True, abstained=True).correct is False


class TestReporting:
    def test_the_headline_table_names_every_measure(self):
        table = headline_table(report([outcome("u1", answerable=False, abstained=True)]))
        for label in ("abstained on unanswerable", "citation precision", "hallucinated"):
            assert label in table

    def test_the_failure_table_says_so_when_there_are_no_failures(self):
        table = failure_table(
            report(
                [
                    outcome("u1", answerable=False, abstained=True),
                    outcome("a1", answerable=True, citations=[1], cited_arxiv=["x"]),
                ]
            )
        )
        assert "No abstention failures" in table

    def test_the_failure_table_lists_the_offending_query_by_id(self):
        table = failure_table(report([outcome("u_bad", answerable=False, abstained=False)]))
        assert "u_bad" in table

    def test_the_json_record_keeps_the_answer_text(self):
        payload = to_dict(
            report([outcome("a1", answerable=True, citations=[1], cited_arxiv=["x"])])
        )
        assert payload["queries"][0]["answer"] == "An answer."
        assert payload["summary"]["answerable"] == 1

    def test_the_markdown_states_what_is_not_measured(self):
        markdown = to_markdown(report([outcome("u1", answerable=False, abstained=True)]))
        assert "not measured" in markdown
        assert "judge model" in markdown
