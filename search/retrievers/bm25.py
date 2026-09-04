"""Lexical retrieval over OpenSearch.

BM25 is the baseline the whole ablation is measured against. If hybrid
retrieval cannot beat a well-configured BM25, the extra machinery is not
earning its place - and a *badly* configured BM25 would make hybrid look good
for the wrong reason, which is the more likely and more embarrassing outcome.

**Why the title is boosted.** A title is a dozen words chosen to describe the
paper; an abstract is two hundred words that include method, motivation and
caveats. BM25 already normalises for field length, but a query term appearing in
the title is stronger evidence than the same term appearing once in an abstract.
``^2`` is the conventional starting point. It is a guess, and Day 9 measures it
rather than leaving it as folklore.

**Why ``best_fields``.** The alternative, ``cross_fields``, treats the fields as
one merged document, which suits queries whose terms are spread across fields -
"surname firstname affiliation". A search query here is normally one coherent
topic, and the best answer is a paper where a *single* field matches it well,
so the best single field should decide the score rather than a sum across
fields.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestion.indexing import SearchIndex, build_client
from search.retrievers.base import (
    DEFAULT_FUSION_DEPTH,
    DEFAULT_TOP_K,
    RetrievalResult,
    collapse_to_papers,
)

logger = logging.getLogger(__name__)

DEFAULT_TITLE_BOOST = 2.0


class BM25Retriever:
    """Keyword retrieval against the ``papers`` alias."""

    name = "bm25"

    def __init__(
        self,
        index: SearchIndex | None = None,
        *,
        title_boost: float = DEFAULT_TITLE_BOOST,
        depth: int = DEFAULT_FUSION_DEPTH,
    ) -> None:
        self._index = index
        self.title_boost = title_boost
        self.depth = depth

    @property
    def index(self) -> SearchIndex:
        """The index, connected on first use.

        Deferred so that constructing a retriever costs nothing. The API builds
        one per request and the harness builds one per configuration; neither
        should open a socket that a cached or empty result set would not need.
        """
        if self._index is None:
            self._index = SearchIndex(build_client())
        return self._index

    def _fields(self) -> list[str]:
        return [f"title^{self.title_boost:g}", "abstract", "text"]

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Ranked papers for ``query``.

        Searches to ``depth`` and truncates afterwards rather than searching to
        ``top_k``. Chunk-to-paper collapsing can only ever shorten the list, so
        asking for exactly ``top_k`` chunks would return fewer than ``top_k``
        papers whenever a multi-chunk paper matched twice.
        """
        if not query.strip():
            return []

        size = max(self.depth, top_k)
        body: dict[str, Any] = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self._fields(),
                    "type": "best_fields",
                }
            },
        }
        hits = self.index.client.search(index=self.index.alias, body=body)["hits"]["hits"]

        candidates = [
            (
                hit["_source"]["arxiv_id"],
                float(hit["_score"]),
                {
                    "retriever": self.name,
                    "chunk_index": hit["_source"].get("chunk_index"),
                    "bm25_score": float(hit["_score"]),
                },
            )
            for hit in hits
        ]
        return collapse_to_papers(candidates)[:top_k]


__all__ = ["DEFAULT_TITLE_BOOST", "BM25Retriever"]
