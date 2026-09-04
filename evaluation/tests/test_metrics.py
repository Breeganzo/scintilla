"""Metric tests worked by hand.

Every expected value below is derived from the definition on paper and written
as a literal. Nothing here calls the implementation to decide what the answer
should be, because a test that computes its expectation the same way the code
does only proves the code is self-consistent - it would pass just as happily
against a metric that is wrong in a way both copies share.

The literals for nDCG were evaluated with ``math.log2`` in a scratch session
and pasted in. The arithmetic is machine-checked; the *derivation* - which
gains land at which ranks, and what the ideal ordering is - is the part done by
hand, and it is the part that goes wrong.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    DEFAULT_KS,
    MetricError,
    aggregate,
    dcg,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

# A single fixed example carried through several metrics, so the tests can be
# compared against each other:
#
#     retrieved:  A  B  C  D  E
#     relevant:      B     D
#     gains:      0  1  0  1  0
RETRIEVED = ["A", "B", "C", "D", "E"]
RELEVANT = ["B", "D"]


class TestRecall:
    """recall@k = (relevant papers inside the top k) / (all relevant papers)."""

    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (1, 0.0),  # top 1 is A, not relevant -> 0/2
            (2, 0.5),  # B found -> 1/2
            (3, 0.5),  # C adds nothing -> 1/2
            (4, 1.0),  # D found -> 2/2
            (5, 1.0),  # already complete
        ],
    )
    def test_the_worked_example(self, k, expected):
        assert recall_at_k(RETRIEVED, RELEVANT, k) == expected

    def test_it_is_capped_by_k_not_by_the_retriever(self):
        """Three relevant papers cannot be more than two-thirds recalled at k=2.

        This ceiling is a property of the golden set. It is the reason recall is
        only ever compared across modes on the same queries, never across
        classes.
        """
        assert recall_at_k(["A", "B"], ["A", "B", "C"], 2) == pytest.approx(2 / 3)

    def test_a_relevant_paper_below_the_cutoff_does_not_count(self):
        assert recall_at_k(["X", "Y", "A"], ["A"], 2) == 0.0
        assert recall_at_k(["X", "Y", "A"], ["A"], 3) == 1.0

    def test_a_retriever_returning_nothing_scores_zero(self):
        assert recall_at_k([], ["A"], 10) == 0.0

    def test_relevant_papers_never_retrieved_still_count_against_it(self):
        """The denominator is the golden set, not what came back.

        If it were the latter, a retriever could improve its recall by returning
        less.
        """
        assert recall_at_k(["A"], ["A", "B", "C", "D"], 10) == 0.25


class TestPrecision:
    """precision@k = (relevant papers inside the top k) / k."""

    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (1, 0.0),  # 0 of 1
            (2, 0.5),  # 1 of 2
            (4, 0.5),  # 2 of 4
            (5, 0.4),  # 2 of 5
        ],
    )
    def test_the_worked_example(self, k, expected):
        assert precision_at_k(RETRIEVED, RELEVANT, k) == expected

    def test_the_denominator_is_k_even_when_fewer_results_came_back(self):
        """Returning two perfect results is not precision 1.0 at k=10.

        Dividing by ``len(retrieved)`` would let a retriever inflate its score by
        withholding results it was less sure about - rewarding timidity, which
        is not the behaviour being measured.
        """
        assert precision_at_k(["A", "B"], ["A", "B"], 10) == 0.2
        assert precision_at_k(["A", "B"], ["A", "B"], 2) == 1.0


class TestReciprocalRank:
    """1 / (position of the first relevant result)."""

    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [
            (["B", "A", "C"], 1.0),  # hit at rank 1
            (["A", "B", "C"], 0.5),  # rank 2
            (["A", "C", "B"], 1 / 3),  # rank 3
            (["A", "C", "E"], 0.0),  # no hit at all
        ],
    )
    def test_only_the_first_hit_matters(self, retrieved, expected):
        assert reciprocal_rank(retrieved, ["B"], 3) == pytest.approx(expected)

    def test_everything_after_the_first_hit_is_ignored(self):
        """Two lists with the same first hit score identically.

        This is MRR's blind spot, stated as a test so it cannot be forgotten
        when reading the table: the second list finds three times as much and
        the metric cannot tell.
        """
        one_hit = reciprocal_rank(["A", "B", "X", "Y"], ["B", "C", "D"], 4)
        three_hits = reciprocal_rank(["A", "B", "C", "D"], ["B", "C", "D"], 4)

        assert one_hit == three_hits == 0.5

    def test_a_hit_beyond_k_does_not_count(self):
        assert reciprocal_rank(["X", "X2", "B"], ["B"], 2) == 0.0
        assert reciprocal_rank(["X", "X2", "B"], ["B"], 3) == pytest.approx(1 / 3)


class TestDCG:
    def test_rank_one_is_undiscounted(self):
        """log2(1 + 1) = 1, so the top position keeps its full gain."""
        assert dcg([1.0]) == 1.0

    def test_the_discount_matches_the_definition(self):
        # rank 2 -> 1/log2(3), rank 3 -> 1/log2(4) = 0.5
        assert dcg([0.0, 1.0]) == pytest.approx(0.6309297535714575)
        assert dcg([0.0, 0.0, 1.0]) == pytest.approx(0.5)

    def test_gains_accumulate(self):
        assert dcg([1.0, 1.0]) == pytest.approx(1.6309297535714575)

    def test_empty(self):
        assert dcg([]) == 0.0


class TestNDCG:
    def test_the_worked_example_at_five(self):
        """gains 0,1,0,1,0 against an ideal of 1,1.

        DCG  = 1/log2(3) + 1/log2(5) = 0.6309297535714575 + 0.43067655807339306
             = 1.0616063116448506
        IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297535714575
             = 1.6309297535714575
        nDCG = 0.6509209298071326
        """
        assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(0.6509209298071326)

    def test_the_worked_example_at_three(self):
        """Only B is inside the cut-off, but the ideal still holds two ones.

        min(len(relevant), k) = min(2, 3) = 2, so the ideal is not truncated
        here and the score drops to 0.38685280723454163. Truncating the ideal to
        the number of hits actually found would make every list its own ideal
        and nDCG would be 1.0 for everything.
        """
        assert ndcg_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.38685280723454163)

    def test_a_perfect_ranking_is_one(self):
        assert ndcg_at_k(["B", "D", "A"], ["B", "D"], 3) == 1.0

    def test_the_ideal_is_truncated_at_k(self):
        """Eight relevant papers and k=2: the best possible is two hits, so 1.0.

        Without truncating the ideal to k this would top out around 0.3 and a
        heavily labelled query could never score well no matter what came back.
        """
        relevant = [f"P{i}" for i in range(8)]
        assert ndcg_at_k(["P0", "P1"], relevant, 2) == 1.0

    def test_order_matters_where_recall_cannot_see_it(self):
        """Same papers, same recall, different nDCG - this is why both are reported.

        For ``late``: gains 0,0,0,1,1, so
        DCG = 1/log2(5) + 1/log2(6) = 0.8175293653079347, over the same ideal
        of 1.6309297535714575.
        """
        early = ["B", "D", "X", "Y", "Z"]
        late = ["X", "Y", "Z", "B", "D"]

        assert recall_at_k(early, RELEVANT, 5) == recall_at_k(late, RELEVANT, 5) == 1.0
        assert ndcg_at_k(early, RELEVANT, 5) == 1.0
        assert ndcg_at_k(late, RELEVANT, 5) == pytest.approx(0.5012658353418871)

    def test_everything_relevant_but_buried(self):
        """Three hits at ranks 8, 9, 10: recall 1.0, nDCG 0.42.

        DCG  = 1/log2(9) + 1/log2(10) + 1/log2(11)
        IDCG = 1 + 1/log2(3) + 1/log2(4)
        """
        retrieved = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "A", "B", "C"]
        relevant = ["A", "B", "C"]

        assert recall_at_k(retrieved, relevant, 10) == 1.0
        assert ndcg_at_k(retrieved, relevant, 10) == pytest.approx(0.42495990177520937)


class TestRefusals:
    """The metrics refuse rather than return a plausible number.

    Each of these inputs has an obvious-looking answer that would be wrong, and
    a wrong number in an ablation table is indistinguishable from a real one.
    """

    @pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k])
    def test_an_empty_relevant_set_is_rejected(self, metric):
        """0/0. Scoring the unanswerable class this way would be meaningless.

        Returning 0.0 punishes a retriever for a query that has no answer;
        returning 1.0 rewards it for returning anything at all. The harness must
        route these queries to an abstention measurement instead.
        """
        with pytest.raises(MetricError, match="[Uu]nanswerable"):
            metric(["A"], [], 10)

    @pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k])
    def test_duplicate_results_are_rejected(self, metric):
        """A paper returned twice would be counted twice.

        ``collapse_to_papers`` prevents this upstream, so a duplicate reaching
        the metric means the collapse was skipped - and recall would read above
        1.0, or worse, just below it.
        """
        with pytest.raises(MetricError, match="duplicates"):
            metric(["A", "A", "B"], ["A"], 10)

    @pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k])
    @pytest.mark.parametrize("k", [0, -1])
    def test_a_nonsensical_cutoff_is_rejected(self, metric, k):
        with pytest.raises(MetricError, match="at least 1"):
            metric(["A"], ["A"], k)


class TestScoreQuery:
    def test_it_reports_every_metric_at_every_cutoff(self):
        scores = score_query(RETRIEVED, RELEVANT)

        expected_keys = {
            f"{name}@{k}" for k in DEFAULT_KS for name in ("recall", "precision", "ndcg")
        }
        assert set(scores) == expected_keys | {"mrr"}

    def test_the_values_match_the_individual_metrics(self):
        scores = score_query(RETRIEVED, RELEVANT, ks=(5,))

        assert scores["recall@5"] == 1.0
        assert scores["precision@5"] == 0.4
        assert scores["ndcg@5"] == pytest.approx(0.6509209298071326)
        assert scores["mrr"] == 0.5

    def test_mrr_is_measured_at_the_deepest_cutoff(self):
        """A hit at rank 8 counts for MRR when k=10 is requested, not when k=5 is."""
        retrieved = [f"X{i}" for i in range(7)] + ["A"]

        assert score_query(retrieved, ["A"], ks=(1, 5))["mrr"] == 0.0
        assert score_query(retrieved, ["A"], ks=(1, 10))["mrr"] == pytest.approx(1 / 8)


class TestAggregate:
    def test_macro_average_counts_every_query_once(self):
        """The query with four labels does not outweigh the query with one.

        Micro-averaging - pooling all hits and all labels before dividing - would
        make the headline number reflect how generously each query happened to be
        labelled rather than how well the retriever did.
        """
        rows = [{"recall@10": 1.0}, {"recall@10": 0.0}, {"recall@10": 0.5}]

        assert aggregate(rows) == {"recall@10": 0.5}

    def test_every_metric_is_averaged(self):
        rows = [{"recall@10": 1.0, "mrr": 1.0}, {"recall@10": 0.0, "mrr": 0.5}]

        assert aggregate(rows) == {"recall@10": 0.5, "mrr": 0.75}

    def test_no_queries_is_an_empty_report_not_a_crash(self):
        """A class can legitimately be empty once the report is filtered."""
        assert aggregate([]) == {}

    def test_mean_of_nothing_is_zero(self):
        assert mean([]) == 0.0
