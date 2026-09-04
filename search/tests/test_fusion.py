"""Tests for Reciprocal Rank Fusion.

**Every expected value here was computed by hand before the code ran.** RRF is
five lines, which is exactly why it deserves this: a sign error, an off-by-one
in the rank, or ``k`` on the wrong side of the division all produce a plausible
ranked list. Nothing downstream would fail - the ablation table would simply
report numbers for an algorithm that is not RRF, and the conclusion drawn from
it would be wrong in a way no test could catch later.

With ``k = 60``:

    rank 1 -> 1/61 = 0.016393442622950821
    rank 2 -> 1/62 = 0.016129032258064516
    rank 3 -> 1/63 = 0.015873015873015872
"""

from __future__ import annotations

import pytest

from search.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from search.tests.conftest import ranked, result

RANK_1 = 1 / 61
RANK_2 = 1 / 62
RANK_3 = 1 / 63


def ids(results) -> list[str]:
    return [item.arxiv_id for item in results]


class TestHandComputed:
    """The values worked out on paper, asserted exactly."""

    def test_the_worked_example(self):
        """bm25 [a, b, c] fused with dense [c, a, d] gives a, c, b, d.

        By hand:
            a = 1/61 + 1/62 = 0.0325224748810153
            c = 1/63 + 1/61 = 0.0322664584959667
            b = 1/62       = 0.0161290322580645
            d = 1/63       = 0.0158730158730159

        The interesting part is a > c. ``c`` is dense's top hit, ``a`` is only
        its second - but ``a`` is ranked above ``c`` by bm25 too, and RRF
        rewards that agreement. A fusion that let either list's rank 1 win
        outright would order these the other way.
        """
        fused = reciprocal_rank_fusion(
            {"bm25": ranked("a", "b", "c"), "dense": ranked("c", "a", "d")}
        )

        assert ids(fused) == ["a", "c", "b", "d"]
        assert fused[0].score == pytest.approx(RANK_1 + RANK_2)
        assert fused[1].score == pytest.approx(RANK_3 + RANK_1)
        assert fused[2].score == pytest.approx(RANK_2)
        assert fused[3].score == pytest.approx(RANK_3)

    def test_agreement_beats_a_single_confident_hit(self):
        """Ranked 3rd by both beats ranked 1st by one and absent from the other.

        2/63 = 0.0317460 against 1/61 = 0.0163934. This is the property that
        makes fusion worth doing, stated as an assertion rather than a claim.
        """
        fused = reciprocal_rank_fusion(
            {
                "bm25": [result("solo", rank=1), result("x", rank=2), result("agreed", rank=3)],
                "dense": [result("y", rank=1), result("z", rank=2), result("agreed", rank=3)],
            }
        )

        assert fused[0].arxiv_id == "agreed"
        assert fused[0].score == pytest.approx(2 * RANK_3)
        assert fused[1].score == pytest.approx(RANK_1)

    def test_k_zero_makes_the_top_rank_dominate(self):
        """At k=0 rank 1 contributes 1.0 and rank 2 only 0.5.

        This is what k damps. Asserting it pins down which side of the division
        k sits on - a sign error would still produce a sorted list.
        """
        fused = reciprocal_rank_fusion({"bm25": ranked("a", "b")}, k=0)

        assert fused[0].score == pytest.approx(1.0)
        assert fused[1].score == pytest.approx(0.5)

    def test_default_k_is_sixty(self):
        assert DEFAULT_RRF_K == 60


class TestEdgeCases:
    def test_document_in_one_list_only(self):
        fused = reciprocal_rank_fusion({"bm25": ranked("a"), "dense": ranked("b")})

        assert {item.arxiv_id for item in fused} == {"a", "b"}
        assert all(item.score == pytest.approx(RANK_1) for item in fused)

    def test_identical_lists_double_every_score_and_keep_the_order(self):
        fused = reciprocal_rank_fusion(
            {"bm25": ranked("a", "b", "c"), "dense": ranked("a", "b", "c")}
        )

        assert ids(fused) == ["a", "b", "c"]
        assert fused[0].score == pytest.approx(2 * RANK_1)
        assert fused[1].score == pytest.approx(2 * RANK_2)
        assert fused[2].score == pytest.approx(2 * RANK_3)

    def test_an_empty_run_contributes_nothing(self):
        fused = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": []})

        assert ids(fused) == ["a", "b"]

    def test_all_runs_empty(self):
        assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []

    def test_no_runs_at_all(self):
        assert reciprocal_rank_fusion({}) == []

    def test_ties_break_on_arxiv_id_so_the_order_is_stable(self):
        """Both at rank 1 in different lists, so both score 1/61 exactly.

        Without a deterministic tie-break the same query could return different
        orderings on different runs, which would surface in evaluation as metric
        drift with no cause.
        """
        first = reciprocal_rank_fusion({"bm25": ranked("zulu"), "dense": ranked("alpha")})
        second = reciprocal_rank_fusion({"dense": ranked("alpha"), "bm25": ranked("zulu")})

        assert ids(first) == ["alpha", "zulu"]
        assert ids(first) == ids(second)


class TestOutputShape:
    def test_ranks_are_contiguous_and_one_based(self):
        fused = reciprocal_rank_fusion(
            {"bm25": ranked("a", "b", "c"), "dense": ranked("d", "e", "f")}
        )

        assert [item.rank for item in fused] == list(range(1, len(fused) + 1))

    def test_top_k_truncates_after_fusion_not_before(self):
        """Truncating first would change which documents win, not just how many."""
        fused = reciprocal_rank_fusion(
            {"bm25": ranked("a", "b", "c"), "dense": ranked("c", "a", "d")}, top_k=2
        )

        assert ids(fused) == ["a", "c"]
        assert [item.rank for item in fused] == [1, 2]

    def test_debug_records_each_retriever_rank_and_contribution(self):
        fused = reciprocal_rank_fusion({"bm25": ranked("a", "b"), "dense": ranked("b", "a")})
        by_id = {item.arxiv_id: item for item in fused}

        runs = by_id["a"].debug["runs"]
        assert runs["bm25"]["rank"] == 1
        assert runs["dense"]["rank"] == 2
        assert runs["bm25"]["contribution"] == pytest.approx(RANK_1)
        assert by_id["a"].debug["rrf_k"] == 60

    def test_debug_shows_when_only_one_retriever_found_a_paper(self):
        fused = reciprocal_rank_fusion({"bm25": ranked("only-lexical"), "dense": []})

        assert list(fused[0].debug["runs"]) == ["bm25"]


class TestInvalidInput:
    def test_zero_rank_is_rejected(self):
        """A 0-based rank would overweight the top hit and never raise on its own."""
        with pytest.raises(ValueError, match="1-based"):
            reciprocal_rank_fusion({"bm25": [result("a", rank=0)]})

    def test_negative_rank_is_rejected(self):
        with pytest.raises(ValueError, match="1-based"):
            reciprocal_rank_fusion({"bm25": [result("a", rank=-1)]})

    def test_negative_k_is_rejected(self):
        """k = -1 would divide by zero at rank 1."""
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion({"bm25": ranked("a")}, k=-1)
