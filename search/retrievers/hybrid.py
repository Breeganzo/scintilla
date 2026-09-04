"""Hybrid retrieval: both retrievers, fused by RRF.

This is the configuration the project is arguing for, and the one the ablation
has to justify. It is deliberately thin - it holds two retrievers, runs both,
and hands the results to :func:`search.fusion.reciprocal_rank_fusion`. All the
interesting behaviour lives in the parts it composes, which is what makes each
of them testable on its own.

**Both retrievers search to full depth, not to ``top_k``.** Fusion can only
reward a document some retriever returned, so cutting each list to 10 would
discard exactly the disagreements RRF exists to resolve. That is not
hypothetical here: the query "learning to rank documents for search engines"
produced BM25 and dense top-3 lists with **no papers in common**. Truncating
early would have thrown away one of those two views entirely.

**A failing retriever degrades the result rather than the request.** If
OpenSearch is down, dense results still answer the query. Returning nothing
because one of two independent systems is unavailable would be a worse outcome
than returning half the evidence, provided the response says which half is
missing - and it does, in ``debug``. What is *not* acceptable is a silent
degradation, because "hybrid quietly became dense-only" is indistinguishable
from "hybrid is no better than dense" in an evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from search.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from search.retrievers.base import (
    DEFAULT_FUSION_DEPTH,
    DEFAULT_TOP_K,
    RetrievalResult,
    Retriever,
)
from search.retrievers.bm25 import BM25Retriever
from search.retrievers.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 and dense retrieval, combined with Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(
        self,
        lexical: Retriever | None = None,
        dense: Retriever | None = None,
        *,
        k: int = DEFAULT_RRF_K,
        depth: int = DEFAULT_FUSION_DEPTH,
        tolerate_failure: bool = True,
    ) -> None:
        self.lexical = lexical or BM25Retriever(depth=depth)
        self.dense = dense or DenseRetriever(depth=depth)
        self.k = k
        self.depth = depth
        self.tolerate_failure = tolerate_failure
        self.degraded: list[str] = []

    def _run(self, retriever: Retriever, query: str) -> list[RetrievalResult]:
        try:
            return retriever.retrieve(query, top_k=self.depth)
        except Exception as exc:  # noqa: BLE001 - recorded and reported, see docstring
            if not self.tolerate_failure:
                raise
            logger.error("Retriever %s failed: %s", retriever.name, exc)
            self.degraded.append(retriever.name)
            return []

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Fused results, best first."""
        if not query.strip():
            return []

        self.degraded = []
        runs: dict[str, list[RetrievalResult]] = {
            self.lexical.name: self._run(self.lexical, query),
            self.dense.name: self._run(self.dense, query),
        }

        if self.degraded and len(self.degraded) == len(runs):
            raise RuntimeError(
                f"Every retriever failed: {', '.join(self.degraded)}. "
                f"Returning an empty result here would look like 'no matches'."
            )

        fused = reciprocal_rank_fusion(runs, k=self.k, top_k=top_k)
        if self.degraded:
            # Mark every result, not just the response envelope. These objects
            # travel into the evaluation harness, and a metric computed from a
            # degraded run must not be mistaken for a metric for hybrid search.
            for result in fused:
                result.debug["degraded"] = list(self.degraded)
        return fused

    def diagnostics(self) -> dict[str, Any]:
        """What the last call did, for the response envelope."""
        return {
            "rrf_k": self.k,
            "fusion_depth": self.depth,
            "degraded": list(self.degraded),
        }


__all__ = ["HybridRetriever"]
