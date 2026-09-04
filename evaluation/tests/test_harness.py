"""Tests for the harness and the report writers.

Everything here runs against stub retrievers with hand-chosen result lists.
That is deliberate: the point of these tests is that the harness scores, groups
and summarises correctly, and a real retriever would make each assertion depend
on the corpus, the model and the index all at once.
"""

from __future__ import annotations

import json

import pytest

from evaluation.golden import GoldenQuery, GoldenSet
from evaluation.harness import ModeReport, QueryOutcome, run_mode, run_query
from evaluation.report import (
    aligned_scores,
    headline_table,
    per_class_table,
    significance_table,
    to_json,
    to_markdown,
    unanswerable_summary,
    where_hybrid_loses,
)
from search.retrievers.base import RetrievalResult


class ListRetriever:
    """Returns a fixed list of arXiv ids per query, ranked in order given."""

    def __init__(self, name: str, by_query: dict[str, list[str]]) -> None:
        self.name = name
        self.by_query = by_query

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        ids = self.by_query.get(query, [])[:top_k]
        return [
            RetrievalResult(arxiv_id=arxiv_id, score=1.0 / rank, rank=rank)
            for rank, arxiv_id in enumerate(ids, start=1)
        ]


def query(qid: str, text: str, klass: str, relevant: tuple[str, ...]) -> GoldenQuery:
    return GoldenQuery(id=qid, query=text, query_class=klass, relevant_ids=relevant)


ANSWERABLE = query("exact_01", "alpha", "exact_term", ("A", "B"))
PARAPHRASE = query("para_01", "beta", "paraphrase", ("C",))
UNANSWERABLE = query("none_01", "gamma", "unanswerable", ())

GOLDEN = GoldenSet(
    version=1,
    corpus={"papers": 10, "chunks": 12},
    pool={"retrievers": ["bm25", "dense"], "depth": 12},
    queries=(ANSWERABLE, PARAPHRASE, UNANSWERABLE),
)


class TestRunQuery:
    def test_it_scores_against_the_labels(self):
        retriever = ListRetriever("stub", {"alpha": ["A", "X", "B"]})

        outcome = run_query(retriever, ANSWERABLE, top_k=10, ks=(1, 10))

        assert outcome.retrieved_ids == ["A", "X", "B"]
        assert outcome.metrics["recall@10"] == 1.0
        assert outcome.metrics["recall@1"] == 0.5
        assert outcome.metrics["mrr"] == 1.0

    def test_an_unanswerable_query_carries_no_ranking_metrics(self):
        """An empty dict, not zeros.

        Zeros would be averaged in and would look exactly like a mode that
        tried and failed, which is a different thing from a query that has no
        correct answer to rank.
        """
        retriever = ListRetriever("stub", {"gamma": ["Q", "R"]})

        outcome = run_query(retriever, UNANSWERABLE, top_k=10, ks=(10,))

        assert outcome.metrics == {}
        assert outcome.retrieved_ids == ["Q", "R"]
        assert outcome.is_unanswerable

    def test_it_records_how_long_retrieval_took(self):
        retriever = ListRetriever("stub", {"alpha": ["A"]})

        assert run_query(retriever, ANSWERABLE, top_k=10, ks=(10,)).elapsed_ms >= 0

    def test_top_k_is_passed_through(self):
        retriever = ListRetriever("stub", {"alpha": ["A", "B", "X"]})

        outcome = run_query(retriever, ANSWERABLE, top_k=2, ks=(2,))

        assert outcome.retrieved_ids == ["A", "B"]


class TestOutcomeAccessors:
    def test_hits_misses_and_first_hit(self):
        outcome = QueryOutcome(
            query_id="q",
            query_class="exact_term",
            query="alpha",
            retrieved_ids=["X", "B", "Y"],
            relevant_ids=["A", "B"],
            metrics={},
            elapsed_ms=1.0,
        )

        assert outcome.hits == ["B"]
        assert outcome.misses == ["A"]
        assert outcome.first_hit_rank == 2

    def test_no_hit_at_all(self):
        outcome = QueryOutcome(
            query_id="q",
            query_class="exact_term",
            query="alpha",
            retrieved_ids=["X"],
            relevant_ids=["A"],
            metrics={},
            elapsed_ms=1.0,
        )

        assert outcome.hits == []
        assert outcome.first_hit_rank is None


class TestRunMode:
    def test_the_unanswerable_query_is_excluded_from_the_averages(self):
        """Forty scored, ten held out - the split the headline number depends on."""
        retriever = ListRetriever("stub", {"alpha": ["A", "B"], "beta": ["C"], "gamma": ["Z"]})

        report = run_mode("bm25", golden=GOLDEN, retriever=retriever, warm_up=False)

        assert len(report.outcomes) == 3
        assert len(report.scored) == 2
        assert len(report.unanswerable) == 1
        assert report.overall["recall@10"] == 1.0

    def test_a_supplied_retriever_is_labelled_as_a_variant(self):
        """A configuration the registry cannot serve must not look like a mode."""
        retriever = ListRetriever("stub", {})

        report = run_mode(
            "hybrid", golden=GOLDEN, retriever=retriever, label="hybrid@k=5", warm_up=False
        )

        assert report.mode == "hybrid@k=5"

    def test_warmup_is_sent_before_anything_is_timed(self):
        """Without it the first query pays for a 12-second model load."""
        seen: list[str] = []

        class Recording(ListRetriever):
            def retrieve(self, query, top_k=10):
                seen.append(query)
                return super().retrieve(query, top_k)

        run_mode("bm25", golden=GOLDEN, retriever=Recording("stub", {}), warm_up=True)

        assert seen[0].startswith("warm up")
        assert len(seen) == 4

    def test_by_class_groups_only_scored_queries(self):
        retriever = ListRetriever("stub", {"alpha": ["A"], "beta": ["C"], "gamma": ["Z"]})

        by_class = run_mode("bm25", golden=GOLDEN, retriever=retriever, warm_up=False).by_class()

        assert set(by_class) == {"exact_term", "paraphrase"}
        assert by_class["paraphrase"]["recall@10"] == 1.0
        assert by_class["exact_term"]["recall@10"] == 0.5

    @pytest.mark.parametrize(
        ("timings", "expected"),
        [([5.0], 5.0), ([1.0, 3.0], 2.0), ([1.0, 5.0, 9.0], 5.0), ([], 0.0)],
    )
    def test_median_latency(self, timings, expected):
        report = ModeReport(
            mode="bm25",
            top_k=10,
            golden_set_version=1,
            outcomes=[
                QueryOutcome("q", "exact_term", "x", [], ["A"], {}, elapsed) for elapsed in timings
            ],
        )

        assert report.median_latency_ms == expected


@pytest.fixture
def reports():
    """Three modes over the same queries, with hybrid deliberately worse."""
    bm25 = run_mode(
        "bm25",
        golden=GOLDEN,
        retriever=ListRetriever("bm25", {"alpha": ["A"], "beta": ["Z"], "gamma": ["Z"]}),
        warm_up=False,
    )
    dense = run_mode(
        "dense",
        golden=GOLDEN,
        retriever=ListRetriever("dense", {"alpha": ["A", "B"], "beta": ["C"], "gamma": ["Z"]}),
        warm_up=False,
    )
    hybrid = run_mode(
        "hybrid",
        golden=GOLDEN,
        retriever=ListRetriever(
            "hybrid", {"alpha": ["X", "A"], "beta": ["Z", "C"], "gamma": ["Z"]}
        ),
        warm_up=False,
    )
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}


class TestTables:
    def test_the_headline_table_has_one_row_per_mode(self, reports):
        lines = headline_table(reports).splitlines()

        assert len(lines) == 5  # header, separator, three modes
        assert lines[0].startswith("| mode |")
        assert "| bm25 |" in lines[2]

    def test_the_class_table_lists_the_classes_in_declared_order(self, reports):
        lines = per_class_table(reports, "recall@10").splitlines()

        assert lines[2].startswith("| exact_term")
        assert lines[3].startswith("| paraphrase")

    def test_unanswerable_summary_counts_what_came_back_anyway(self, reports):
        """No retriever abstains, which is the case for an answering layer."""
        text = unanswerable_summary(reports)

        assert "| bm25 | 1 | 1 | 0 |" in text


class TestWhereHybridLoses:
    def test_it_names_the_query_and_the_winner(self, reports):
        losses = where_hybrid_loses(reports, "ndcg@10")

        assert [row["query_id"] for row in losses] == ["exact_01", "para_01"]
        assert all(row["beaten_by"] == "dense" for row in losses)

    def test_it_is_sorted_by_deficit(self, reports):
        deficits = [row["deficit"] for row in where_hybrid_loses(reports, "ndcg@10")]

        assert deficits == sorted(deficits, reverse=True)

    def test_it_records_what_was_missed(self, reports):
        losses = {row["query_id"]: row for row in where_hybrid_loses(reports, "ndcg@10")}

        assert losses["exact_01"]["missed"] == ["B"]

    def test_no_hybrid_means_nothing_to_report(self, reports):
        assert where_hybrid_loses({"bm25": reports["bm25"]}) == []


class TestSignificanceTable:
    def test_it_compares_every_pair(self, reports):
        lines = significance_table(reports).splitlines()

        assert len(lines) == 5
        assert "dense - bm25" in lines[2]
        assert "hybrid - bm25" in lines[3]
        assert "hybrid - dense" in lines[4]

    def test_modes_scored_on_different_queries_are_rejected(self, reports):
        """Pairing two unaligned lists produces a confidence interval for nothing."""
        other = run_mode(
            "bm25",
            golden=GoldenSet(
                version=1, corpus={}, pool={}, queries=(query("zzz_99", "d", "exact_term", ("A",)),)
            ),
            retriever=ListRetriever("stub", {}),
            warm_up=False,
        )

        with pytest.raises(ValueError, match="different queries"):
            significance_table({"a": reports["bm25"], "b": other})

    def test_aligned_scores_are_sorted_by_query_id(self, reports):
        ids, _ = aligned_scores(reports["bm25"], "ndcg@10")

        assert ids == ["exact_01", "para_01"]


class TestSerialisation:
    def test_json_keeps_every_query_not_just_the_average(self, reports):
        payload = json.loads(to_json(reports, {"git_sha": "abc123"}))

        assert payload["meta"]["git_sha"] == "abc123"
        assert len(payload["modes"]["bm25"]["queries"]) == 3
        assert payload["modes"]["bm25"]["queries"][0]["retrieved_ids"] == ["A"]

    def test_json_includes_the_unanswerable_queries(self, reports):
        """Held out of the averages, not dropped from the record."""
        payload = json.loads(to_json(reports))
        ids = [row["query_id"] for row in payload["modes"]["dense"]["queries"]]

        assert "none_01" in ids

    def test_markdown_states_the_scored_and_held_out_counts(self, reports):
        text = to_markdown(reports, {"corpus_papers": 10, "corpus_chunks": 12})

        assert "2 answerable queries scored" in text
        assert "1 unanswerable queries held out" in text
        assert "Where hybrid loses" in text

    def test_markdown_says_so_when_hybrid_wins_everywhere(self, reports):
        text = to_markdown({"dense": reports["dense"], "hybrid": reports["dense"]})

        assert "at least as good as both baselines" in text
