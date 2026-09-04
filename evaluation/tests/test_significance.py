"""Tests for the paired bootstrap.

The properties checked here are the ones that make a confidence interval
trustworthy rather than decorative: it must contain zero when the modes are
equivalent, exclude zero when they are plainly not, respect the pairing between
queries, and produce the same number every time it is run.
"""

from __future__ import annotations

import pytest

from evaluation.significance import compare

# Small iteration counts keep the suite fast. The properties being asserted are
# robust to resampling noise; nothing here depends on a third decimal place.
FAST = 2000


class TestNoDifference:
    def test_identical_scores_give_a_zero_difference(self):
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]

        result = compare(scores, scores, iterations=FAST)

        assert result.difference == 0.0
        assert result.wins_a == result.wins_b == 0
        assert result.ties == 5

    def test_identical_scores_are_not_significant(self):
        """A zero gap must never be reported as a win."""
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]

        assert not compare(scores, scores, iterations=FAST).significant

    def test_a_tiny_gap_over_few_queries_is_undecided(self):
        """This is the case the module exists for.

        Mode b wins narrowly on average but loses on nearly half the queries.
        Reporting that as an improvement is the mistake; the interval spans zero
        and the honest answer is "cannot tell from this many queries".
        """
        a = [0.4, 0.9, 0.2, 0.7, 0.5, 0.8, 0.1, 0.6]
        b = [0.5, 0.8, 0.3, 0.6, 0.6, 0.7, 0.2, 0.7]

        result = compare(a, b, iterations=FAST)

        assert result.difference > 0
        assert result.ci_low < 0 < result.ci_high
        assert not result.significant


class TestRealDifference:
    def test_a_consistent_offset_is_detected(self):
        a = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.15]
        b = [value + 0.3 for value in a]

        result = compare(a, b, iterations=FAST)

        assert result.difference == pytest.approx(0.3)
        assert result.ci_low > 0
        assert result.significant
        assert result.wins_b == 10

    def test_the_interval_brackets_the_observed_difference(self):
        a = [0.1, 0.4, 0.2, 0.9, 0.3, 0.6, 0.5, 0.2]
        b = [0.6, 0.7, 0.9, 0.95, 0.8, 0.85, 0.9, 0.7]

        result = compare(a, b, iterations=FAST)

        assert result.ci_low <= result.difference <= result.ci_high

    def test_direction_is_reported_by_sign(self):
        """``compare(a, b)`` measures b minus a, so a worse b is negative."""
        a = [0.8, 0.9, 0.7, 0.85]
        b = [0.2, 0.3, 0.1, 0.25]

        assert compare(a, b, iterations=FAST).difference < 0
        assert compare(b, a, iterations=FAST).difference > 0


class TestPairing:
    def test_shuffling_one_side_changes_the_answer(self):
        """Proof the test is paired rather than comparing two independent means.

        Both orderings have identical means, so an unpaired test would return
        the same interval for each. A paired test must not, because the pairing
        is where most of the information is.
        """
        a = [0.1, 0.2, 0.8, 0.9]
        aligned = [0.2, 0.3, 0.9, 1.0]
        shuffled = [1.0, 0.9, 0.3, 0.2]

        tight = compare(a, aligned, iterations=FAST)
        loose = compare(a, shuffled, iterations=FAST)

        assert tight.difference == pytest.approx(loose.difference)
        assert (tight.ci_high - tight.ci_low) < (loose.ci_high - loose.ci_low)

    def test_mismatched_lengths_are_rejected(self):
        """Different query counts means the modes were not scored on the same set."""
        with pytest.raises(ValueError, match="same queries"):
            compare([0.1, 0.2], [0.1], iterations=FAST)

    def test_nothing_to_compare_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to compare"):
            compare([], [], iterations=FAST)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_interval(self):
        """A published interval that moves between runs invites re-rolling it."""
        a = [0.1, 0.5, 0.3, 0.8, 0.2, 0.6]
        b = [0.4, 0.6, 0.5, 0.7, 0.5, 0.9]

        first = compare(a, b, iterations=FAST)
        second = compare(a, b, iterations=FAST)

        assert (first.ci_low, first.ci_high, first.p_value) == (
            second.ci_low,
            second.ci_high,
            second.p_value,
        )

    def test_a_different_seed_gives_a_similar_interval(self):
        """Sanity check that the result is a property of the data, not the seed."""
        a = [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7]
        b = [value + 0.2 for value in a]

        first = compare(a, b, iterations=FAST, seed=1)
        second = compare(a, b, iterations=FAST, seed=2)

        assert first.ci_low == pytest.approx(second.ci_low, abs=0.05)
        assert first.ci_high == pytest.approx(second.ci_high, abs=0.05)


class TestPValue:
    def test_it_is_never_exactly_zero(self):
        """10,000 resamples cannot support a claim stronger than p < 1e-4."""
        a = [0.0] * 12
        b = [1.0] * 12

        result = compare(a, b, iterations=FAST)

        assert result.p_value > 0
        assert result.p_value < 0.01

    def test_it_is_high_when_the_modes_are_equivalent(self):
        scores = [0.3, 0.6, 0.1, 0.9, 0.5]

        assert compare(scores, scores, iterations=FAST).p_value > 0.5


class TestDescribe:
    def test_it_states_the_verdict_in_words(self):
        a = [0.1] * 10
        b = [0.9] * 10

        text = compare(a, b, iterations=FAST).describe("bm25", "dense")

        assert "dense - bm25" in text
        assert "significant" in text
        assert "n=10" in text

    def test_an_undecided_result_says_so(self):
        a = [0.4, 0.9, 0.2, 0.7, 0.5, 0.8, 0.1, 0.6]
        b = [0.5, 0.8, 0.3, 0.6, 0.6, 0.7, 0.2, 0.7]

        assert "not distinguishable" in compare(a, b, iterations=FAST).describe()
