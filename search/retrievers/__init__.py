"""Retriever implementations, and the registry that resolves a mode name.

**The registry is the ablation mechanism.** The API accepts ``mode`` and the
evaluation harness will pass the same three strings. One lookup table means the
harness cannot measure a retriever the API cannot serve, and the API cannot
serve one the harness never measured. Two lists of modes that have to be kept in
agreement is precisely how an ablation ends up describing code nobody runs.
"""

from __future__ import annotations

from search.retrievers.base import (
    DEFAULT_FUSION_DEPTH,
    DEFAULT_TOP_K,
    RetrievalResult,
    Retriever,
    collapse_to_papers,
    rerank,
)
from search.retrievers.bm25 import BM25Retriever
from search.retrievers.dense import DenseRetriever
from search.retrievers.hybrid import HybridRetriever

# Ordered so that the API schema, the docs and the ablation table all present
# the modes the same way: the two baselines, then the thing being argued for.
RETRIEVERS: dict[str, type] = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
}

MODES = tuple(RETRIEVERS)
DEFAULT_MODE = "hybrid"


class UnknownModeError(ValueError):
    """Raised for a mode that has no implementation."""


def get_retriever(mode: str = DEFAULT_MODE, **kwargs) -> Retriever:
    """Build the retriever registered under ``mode``."""
    try:
        factory = RETRIEVERS[mode]
    except KeyError as exc:
        raise UnknownModeError(
            f"Unknown retrieval mode {mode!r}. Available: {', '.join(MODES)}"
        ) from exc
    return factory(**kwargs)


__all__ = [
    "DEFAULT_FUSION_DEPTH",
    "DEFAULT_MODE",
    "DEFAULT_TOP_K",
    "MODES",
    "RETRIEVERS",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "Retriever",
    "UnknownModeError",
    "collapse_to_papers",
    "get_retriever",
    "rerank",
]
