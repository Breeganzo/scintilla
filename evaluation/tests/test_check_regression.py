"""Tests for the regression gate's decision logic.

The gate itself is only as trustworthy as the comparison in the middle of it,
and that comparison is a pure function of two dictionaries - so it is tested
directly, without seeding a corpus. A gate whose verdict is untested is a gate
that can silently start passing everything, which is worse than no gate because
it looks like coverage.
"""

from __future__ import annotations

from evaluation.management.commands.check_regression import (
    DEFAULT_TOLERANCE,
    compare,
    regressed_queries,
)


def run(
    recall5: float = 0.5,
    recall10: float = 0.7,
    mrr: float = 0.6,
    ndcg: float = 0.65,
    **provenance,
) -> dict:
    base = {
        "papers": 340,
        "chunks": 340,
        "golden_set_version": 1,
        "queries": 20,
        "top_k": 10,
        "overall": {
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr": mrr,
            "ndcg@10": ndcg,
        },
        "per_query": {},
    }
    base.update(provenance)
    return base


class TestNoChange:
    def test_identical_runs_pass(self):
        failures, notes = compare(run(), run())
        assert failures == []
        assert len(notes) == 4

    def test_an_improvement_passes(self):
        failures, _ = compare(run(), run(recall10=0.9))
        assert failures == []


class TestTolerance:
    def test_a_drop_inside_the_tolerance_passes(self):
        failures, _ = compare(run(), run(ndcg=0.65 - DEFAULT_TOLERANCE + 0.001))
        assert failures == []

    def test_a_drop_beyond_the_tolerance_fails(self):
        failures, _ = compare(run(), run(ndcg=0.65 - DEFAULT_TOLERANCE - 0.001))
        assert len(failures) == 1
        assert "ndcg@10" in failures[0]

    def test_the_tolerance_is_configurable(self):
        failures, _ = compare(run(), run(ndcg=0.60), tolerance=0.10)
        assert failures == []

    def test_each_metric_is_judged_on_its_own(self):
        # A collapse in recall must not be able to hide behind a gain in MRR.
        failures, _ = compare(run(), run(recall10=0.2, mrr=0.99))
        assert len(failures) == 1
        assert "recall@10" in failures[0]


class TestProvenance:
    def test_a_different_corpus_size_is_not_treated_as_a_pass(self):
        # Even with better scores: a larger corpus is a different experiment.
        failures, _ = compare(run(), run(recall10=0.99, papers=400))
        assert any("papers changed" in f for f in failures)

    def test_a_different_golden_set_version_fails(self):
        failures, _ = compare(run(), run(golden_set_version=2))
        assert any("golden_set_version" in f for f in failures)

    def test_a_different_top_k_fails(self):
        failures, _ = compare(run(), run(top_k=20))
        assert any("top_k" in f for f in failures)

    def test_the_message_says_how_to_fix_it(self):
        failures, _ = compare(run(), run(chunks=999))
        assert "regenerate the baseline" in failures[0]


class TestMissingMetrics:
    def test_a_metric_absent_from_the_new_run_fails(self):
        current = run()
        del current["overall"]["mrr"]
        failures, _ = compare(run(), current)
        assert any("mrr is missing" in f for f in failures)


class TestRegressedQueries:
    def test_only_queries_that_got_worse_are_listed(self):
        before = run()
        after = run()
        before["per_query"] = {
            "exact_01": {"ndcg@10": 0.9},
            "exact_02": {"ndcg@10": 0.5},
            "hop_01": {"ndcg@10": 0.3},
        }
        after["per_query"] = {
            "exact_01": {"ndcg@10": 0.4},
            "exact_02": {"ndcg@10": 0.5},
            "hop_01": {"ndcg@10": 0.8},
        }
        worse = regressed_queries(before, after)
        assert len(worse) == 1
        assert "exact_01" in worse[0]

    def test_an_unchanged_query_is_not_reported(self):
        before = run()
        after = run()
        before["per_query"] = {"exact_01": {"ndcg@10": 0.5}}
        after["per_query"] = {"exact_01": {"ndcg@10": 0.5}}
        assert regressed_queries(before, after) == []
