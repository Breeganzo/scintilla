"""Dense retrieval over pgvector.

**This queries PostgreSQL, not OpenSearch.** The original design put the vector
index in OpenSearch alongside the lexical one; that plan did not survive Day 5.
OpenSearch publishes no macOS distribution, the Homebrew formula compiles the
minimal build, and the k-NN plugin ships native libraries for Linux and Windows
only. The full reasoning, and what it costs, is in ``ingestion.vectors``.

Fusion is unaffected: RRF combines ranked lists from independent systems by
construction, so two stores is its normal case rather than a compromise of it.

**The query prefix is the detail most likely to be got wrong here.** BGE models
are trained asymmetrically - queries carry an instruction prefix, documents do
not. Applying it to both sides, or to neither, does not raise, log or fail a
test. It just makes retrieval quietly worse. ``embed_query`` owns the prefix
precisely so no caller can forget it, and there is a test asserting queries and
documents are embedded differently, because a refactor that "simplified" the two
methods into one would be invisible without it.
"""

from __future__ import annotations

import logging

from ingestion.embedding import Embedder, get_embedder
from ingestion.vectors import knn_search
from search.retrievers.base import (
    DEFAULT_FUSION_DEPTH,
    DEFAULT_TOP_K,
    RetrievalResult,
    collapse_to_papers,
)

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Nearest-neighbour retrieval over chunk embeddings."""

    name = "dense"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        depth: int = DEFAULT_FUSION_DEPTH,
    ) -> None:
        self._embedder = embedder
        self.depth = depth

    @property
    def embedder(self) -> Embedder:
        """The embedder, loaded on first use.

        Constructing this class must stay free: it happens per request, and
        loading a transformer model takes seconds. The model itself is cached
        process-wide inside ``ingestion.embedding``, so the cost is paid once
        per worker rather than once per query.
        """
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Ranked papers whose chunk embeddings sit nearest the query vector.

        Searches to ``depth`` before collapsing chunks to papers, for the same
        reason BM25 does: collapsing shortens the list, so retrieving exactly
        ``top_k`` chunks can return fewer than ``top_k`` papers.
        """
        if not query.strip():
            return []

        vector = self.embedder.embed_query(query)
        size = max(self.depth, top_k)
        rows = knn_search(vector, size=size)

        candidates = [
            (
                chunk.paper.arxiv_id,
                similarity,
                {
                    "retriever": self.name,
                    "chunk_index": chunk.chunk_index,
                    "similarity": similarity,
                    "embedding_model": chunk.embedding_model,
                },
            )
            for chunk, similarity in rows
        ]
        return collapse_to_papers(candidates)[:top_k]


__all__ = ["DenseRetriever"]
