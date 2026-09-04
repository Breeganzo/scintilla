"""Tests for the retrieval interface: result shape, chunk collapsing, reranking."""

from __future__ import annotations

from search.retrievers.base import (
    DEFAULT_FUSION_DEPTH,
    RetrievalResult,
    Retriever,
    collapse_to_papers,
    rerank,
)
from search.retrievers.bm25 import BM25Retriever
from search.retrievers.dense import DenseRetriever
from search.retrievers.hybrid import HybridRetriever
from search.tests.conftest import ranked


class TestProtocolConformance:
    def test_every_retriever_satisfies_the_protocol(self):
        """A wiring mistake should fail here, not as an AttributeError in a view."""
        assert isinstance(BM25Retriever(), Retriever)
        assert isinstance(DenseRetriever(), Retriever)
        assert isinstance(HybridRetriever(), Retriever)

    def test_every_retriever_has_a_distinct_name(self):
        names = {BM25Retriever.name, DenseRetriever.name, HybridRetriever.name}

        assert names == {"bm25", "dense", "hybrid"}

    def test_fusion_depth_exceeds_a_typical_top_k(self):
        """Fusion needs depth to work with; see the constant's comment."""
        assert DEFAULT_FUSION_DEPTH >= 50


class TestCollapseToPapers:
    def test_one_chunk_per_paper_passes_through_in_order(self):
        collapsed = collapse_to_papers([("a", 9.0, {}), ("b", 8.0, {}), ("c", 7.0, {})])

        assert [item.arxiv_id for item in collapsed] == ["a", "b", "c"]
        assert [item.rank for item in collapsed] == [1, 2, 3]

    def test_a_paper_matching_twice_is_counted_once(self):
        """Otherwise a long paper collects one RRF contribution per chunk."""
        collapsed = collapse_to_papers(
            [("a", 9.0, {"chunk_index": 0}), ("a", 8.5, {"chunk_index": 1}), ("b", 8.0, {})]
        )

        assert [item.arxiv_id for item in collapsed] == ["a", "b"]

    def test_the_best_ranked_chunk_wins(self):
        collapsed = collapse_to_papers(
            [("a", 9.0, {"chunk_index": 3}), ("a", 8.5, {"chunk_index": 1})]
        )

        assert collapsed[0].score == 9.0
        assert collapsed[0].debug["chunk_index"] == 3

    def test_extra_chunk_hits_are_recorded_not_discarded(self):
        """Useful when a result's position looks wrong; it must not affect scoring."""
        collapsed = collapse_to_papers([("a", 9.0, {}), ("a", 8.0, {}), ("a", 7.0, {})])

        assert collapsed[0].debug["extra_chunk_hits"] == 2

    def test_ranks_stay_contiguous_after_collapsing(self):
        """A gap here would silently change every fused score below it."""
        collapsed = collapse_to_papers(
            [("a", 9.0, {}), ("a", 8.0, {}), ("b", 7.0, {}), ("b", 6.0, {}), ("c", 5.0, {})]
        )

        assert [item.rank for item in collapsed] == [1, 2, 3]

    def test_empty_input(self):
        assert collapse_to_papers([]) == []

    def test_debug_dicts_are_copied_not_shared(self):
        """Two results sharing one dict would let a mutation leak between them."""
        shared = {"retriever": "bm25"}
        collapsed = collapse_to_papers([("a", 1.0, shared), ("b", 1.0, shared)])
        collapsed[0].debug["mutated"] = True

        assert "mutated" not in collapsed[1].debug
        assert "mutated" not in shared


class TestRerank:
    def test_renumbers_from_one(self):
        reranked = rerank([RetrievalResult("a", 1.0, 5), RetrievalResult("b", 0.5, 9)])

        assert [item.rank for item in reranked] == [1, 2]

    def test_preserves_everything_else(self):
        original = RetrievalResult("a", 1.5, 7, {"retriever": "bm25"})
        reranked = rerank([original])[0]

        assert reranked.arxiv_id == "a"
        assert reranked.score == 1.5
        assert reranked.debug == {"retriever": "bm25"}

    def test_does_not_mutate_the_input(self):
        original = ranked("a", "b")
        rerank(list(reversed(original)))

        assert [item.rank for item in original] == [1, 2]

    def test_empty_list(self):
        assert rerank([]) == []
