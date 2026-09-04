"""Tests for the three retrievers.

None of these touch OpenSearch or load a transformer model. What is under test
is ranking behaviour and query construction, and both are decisions this
codebase makes - not behaviour of the services it calls. Tests that needed a
live cluster would be testing OpenSearch, would fail on a machine with no
network, and would be slow enough that nobody ran them.
"""

from __future__ import annotations

import pytest

from ingestion.embedding import QUERY_PREFIX
from search.retrievers.bm25 import DEFAULT_TITLE_BOOST, BM25Retriever
from search.retrievers.dense import DenseRetriever
from search.retrievers.hybrid import HybridRetriever
from search.tests.conftest import (
    FakeSearchIndex,
    RecordingEmbedder,
    StubRetriever,
    fake_chunk,
    hit,
    ranked,
)


class TestBM25Retriever:
    def test_returns_hits_in_opensearch_order(self):
        index = FakeSearchIndex([hit("a", 14.2), hit("b", 9.1), hit("c", 3.0)])

        results = BM25Retriever(index).retrieve("higgs boson")

        assert [item.arxiv_id for item in results] == ["a", "b", "c"]
        assert [item.rank for item in results] == [1, 2, 3]

    def test_carries_the_opensearch_score(self):
        index = FakeSearchIndex([hit("a", 14.25)])

        results = BM25Retriever(index).retrieve("higgs")

        assert results[0].score == pytest.approx(14.25)
        assert results[0].debug["bm25_score"] == pytest.approx(14.25)

    def test_boosts_the_title(self):
        """A boost that silently disappeared would still return a sorted list."""
        index = FakeSearchIndex([hit("a", 1.0)])

        BM25Retriever(index).retrieve("higgs")

        fields = index.client.last_body["query"]["multi_match"]["fields"]
        assert fields[0] == f"title^{DEFAULT_TITLE_BOOST:g}"
        assert "abstract" in fields

    def test_the_boost_is_configurable(self):
        index = FakeSearchIndex([hit("a", 1.0)])

        BM25Retriever(index, title_boost=3).retrieve("higgs")

        assert index.client.last_body["query"]["multi_match"]["fields"][0] == "title^3"

    def test_uses_best_fields(self):
        index = FakeSearchIndex([hit("a", 1.0)])

        BM25Retriever(index).retrieve("higgs")

        assert index.client.last_body["query"]["multi_match"]["type"] == "best_fields"

    def test_queries_the_alias_never_a_concrete_index(self):
        """Querying a concrete index would break silently after a rebuild."""
        index = FakeSearchIndex([hit("a", 1.0)], alias="papers")

        BM25Retriever(index).retrieve("higgs")

        assert index.client.last_index == "papers"

    def test_searches_to_fusion_depth_not_to_top_k(self):
        """Collapsing chunks shortens the list, so top_k chunks is too few."""
        index = FakeSearchIndex([hit("a", 1.0)])

        BM25Retriever(index, depth=50).retrieve("higgs", top_k=5)

        assert index.client.last_body["size"] == 50

    def test_truncates_to_top_k_after_collapsing(self):
        index = FakeSearchIndex([hit(str(n), float(50 - n)) for n in range(20)])

        results = BM25Retriever(index).retrieve("higgs", top_k=5)

        assert len(results) == 5

    def test_collapses_multiple_chunks_of_one_paper(self):
        index = FakeSearchIndex(
            [hit("a", 9.0, chunk_index=0), hit("a", 8.0, chunk_index=1), hit("b", 7.0)]
        )

        results = BM25Retriever(index).retrieve("higgs")

        assert [item.arxiv_id for item in results] == ["a", "b"]

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_blank_queries_do_not_reach_opensearch(self, query):
        index = FakeSearchIndex([hit("a", 1.0)])

        assert BM25Retriever(index).retrieve(query) == []
        assert index.client.last_body is None

    def test_no_hits(self):
        assert BM25Retriever(FakeSearchIndex([])).retrieve("higgs") == []


class TestDenseRetriever:
    def test_applies_the_bge_query_prefix(self, monkeypatch):
        """The prefix is invisible in the results and absent from any error.

        Its absence costs retrieval quality and nothing else, so capturing the
        text that was actually encoded is the only way a refactor that drops it
        gets caught.
        """
        embedder = RecordingEmbedder()
        monkeypatch.setattr("search.retrievers.dense.knn_search", lambda *a, **k: [])

        DenseRetriever(embedder).retrieve("what causes neutrino oscillation")

        assert embedder.queries == ["what causes neutrino oscillation"]

    def test_documents_and_queries_are_embedded_differently(self):
        """BGE is trained asymmetrically. Collapsing the two methods is a silent regression."""
        from ingestion.embedding import HashingEmbedder

        embedder = HashingEmbedder(dimension=64)
        text = "neutrino oscillation"

        assert embedder.embed_query(text) != embedder.embed_documents([text])[0]
        assert QUERY_PREFIX.strip()

    def test_returns_papers_ordered_by_similarity(self, monkeypatch):
        rows = [
            (fake_chunk("a"), 0.91),
            (fake_chunk("b"), 0.84),
            (fake_chunk("c"), 0.77),
        ]
        monkeypatch.setattr("search.retrievers.dense.knn_search", lambda *a, **k: rows)

        results = DenseRetriever(RecordingEmbedder()).retrieve("neutrinos")

        assert [item.arxiv_id for item in results] == ["a", "b", "c"]
        assert [item.rank for item in results] == [1, 2, 3]

    def test_similarity_is_carried_as_the_score(self, monkeypatch):
        """Higher must mean better. knn_search converts distance for this reason."""
        monkeypatch.setattr(
            "search.retrievers.dense.knn_search", lambda *a, **k: [(fake_chunk("a"), 0.91)]
        )

        results = DenseRetriever(RecordingEmbedder()).retrieve("neutrinos")

        assert results[0].score == pytest.approx(0.91)
        assert results[0].debug["similarity"] == pytest.approx(0.91)

    def test_records_which_model_produced_the_vector(self, monkeypatch):
        """A mixed-model index returns meaningless neighbours without erroring."""
        monkeypatch.setattr(
            "search.retrievers.dense.knn_search",
            lambda *a, **k: [(fake_chunk("a", model="bge-small"), 0.9)],
        )

        results = DenseRetriever(RecordingEmbedder()).retrieve("neutrinos")

        assert results[0].debug["embedding_model"] == "bge-small"

    def test_searches_to_fusion_depth(self, monkeypatch):
        captured = {}

        def spy(vector, *, size, **kwargs):
            captured["size"] = size
            return []

        monkeypatch.setattr("search.retrievers.dense.knn_search", spy)
        DenseRetriever(RecordingEmbedder(), depth=50).retrieve("neutrinos", top_k=5)

        assert captured["size"] == 50

    def test_collapses_multiple_chunks_of_one_paper(self, monkeypatch):
        rows = [(fake_chunk("a", 0), 0.9), (fake_chunk("a", 1), 0.88), (fake_chunk("b"), 0.7)]
        monkeypatch.setattr("search.retrievers.dense.knn_search", lambda *a, **k: rows)

        results = DenseRetriever(RecordingEmbedder()).retrieve("neutrinos")

        assert [item.arxiv_id for item in results] == ["a", "b"]

    @pytest.mark.parametrize("query", ["", "   "])
    def test_blank_queries_do_not_load_the_model(self, query):
        """Embedding a blank string would cost a model load to return nothing."""
        embedder = RecordingEmbedder()

        assert DenseRetriever(embedder).retrieve(query) == []
        assert embedder.queries == []


class TestHybridRetriever:
    def test_fuses_both_runs(self):
        lexical = StubRetriever("bm25", ranked("a", "b", "c"))
        dense = StubRetriever("dense", ranked("c", "a", "d"))

        results = HybridRetriever(lexical, dense).retrieve("higgs", top_k=4)

        assert [item.arxiv_id for item in results] == ["a", "c", "b", "d"]

    def test_asks_both_retrievers_for_full_depth(self):
        """Truncating to top_k first would discard the disagreements fusion resolves."""
        lexical = StubRetriever("bm25", ranked("a"))
        dense = StubRetriever("dense", ranked("b"))

        HybridRetriever(lexical, dense, depth=50).retrieve("higgs", top_k=3)

        assert lexical.calls == [("higgs", 50)]
        assert dense.calls == [("higgs", 50)]

    def test_a_paper_found_by_both_outranks_one_found_by_either(self):
        lexical = StubRetriever("bm25", ranked("solo-lex", "shared"))
        dense = StubRetriever("dense", ranked("solo-dense", "shared"))

        results = HybridRetriever(lexical, dense).retrieve("higgs")

        assert results[0].arxiv_id == "shared"

    def test_one_retriever_failing_degrades_rather_than_fails(self):
        """Half the evidence beats none, provided the response says which half."""
        lexical = StubRetriever("bm25", error=ConnectionError("OpenSearch is down"))
        dense = StubRetriever("dense", ranked("a", "b"))

        hybrid = HybridRetriever(lexical, dense)
        results = hybrid.retrieve("higgs")

        assert [item.arxiv_id for item in results] == ["a", "b"]
        assert hybrid.diagnostics()["degraded"] == ["bm25"]

    def test_degradation_is_marked_on_every_result(self):
        """These objects reach the evaluation harness; a degraded run must be visible."""
        lexical = StubRetriever("bm25", error=ConnectionError("down"))
        dense = StubRetriever("dense", ranked("a"))

        results = HybridRetriever(lexical, dense).retrieve("higgs")

        assert results[0].debug["degraded"] == ["bm25"]

    def test_both_retrievers_failing_raises(self):
        """An empty list here is indistinguishable from 'no matches'."""
        lexical = StubRetriever("bm25", error=ConnectionError("down"))
        dense = StubRetriever("dense", error=ConnectionError("down"))

        with pytest.raises(RuntimeError, match="Every retriever failed"):
            HybridRetriever(lexical, dense).retrieve("higgs")

    def test_failures_can_be_made_fatal_for_the_evaluation_harness(self):
        """Phase 3 must never score a degraded run as if it were hybrid search."""
        lexical = StubRetriever("bm25", error=ConnectionError("down"))
        dense = StubRetriever("dense", ranked("a"))

        with pytest.raises(ConnectionError):
            HybridRetriever(lexical, dense, tolerate_failure=False).retrieve("higgs")

    def test_diagnostics_report_the_fusion_parameters(self):
        hybrid = HybridRetriever(StubRetriever("bm25"), StubRetriever("dense"), k=42, depth=25)
        hybrid.retrieve("higgs")

        assert hybrid.diagnostics() == {"rrf_k": 42, "fusion_depth": 25, "degraded": []}

    def test_degraded_state_resets_between_calls(self):
        """Stale state would mark a healthy run as degraded."""
        lexical = StubRetriever("bm25", error=ConnectionError("down"))
        dense = StubRetriever("dense", ranked("a"))
        hybrid = HybridRetriever(lexical, dense)

        hybrid.retrieve("first")
        lexical.error = None
        lexical.results = ranked("b")
        hybrid.retrieve("second")

        assert hybrid.diagnostics()["degraded"] == []

    @pytest.mark.parametrize("query", ["", "  "])
    def test_blank_queries_reach_neither_retriever(self, query):
        lexical = StubRetriever("bm25", ranked("a"))
        dense = StubRetriever("dense", ranked("b"))

        assert HybridRetriever(lexical, dense).retrieve(query) == []
        assert lexical.calls == []
