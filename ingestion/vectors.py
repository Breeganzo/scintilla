"""Dense retrieval, in PostgreSQL, using pgvector.

**Why the vectors are not in OpenSearch.**

The original design put both retrieval modes in OpenSearch: BM25 over analysed
text and kNN over a ``knn_vector`` field. That plan did not survive contact with
the development machine. OpenSearch publishes no macOS distribution at all - the
Homebrew formula compiles the *minimal* build from source, and the k-NN plugin
ships native JNI libraries that exist only for Linux and Windows. Docker, which
would normally paper over exactly this, is not available on this laptop either.

The options were: fake the vector store in development and hope the real one
behaves the same, develop only against the deployment VM, or move dense
retrieval somewhere that runs natively in both places. The first two are how you
get a system that works on one machine.

pgvector turns out to be a better answer than a workaround, for three reasons:

1. **The vector and the row it describes are written in one transaction.** The
   original design had to track whether PostgreSQL and OpenSearch agreed, and
   accept that they sometimes would not. Half of that consistency problem
   disappears when the vector lives next to its text.
2. **One fewer moving part on a 1 GB VM.** The dense index costs a column and a
   PostgreSQL index rather than a second JVM holding a second copy of the
   corpus.
3. **Hybrid retrieval is unaffected.** Reciprocal Rank Fusion combines *ranked
   lists*, not scores, and was designed to fuse runs from entirely separate
   systems. Two stores is the normal case for RRF, not a compromise of it.

What is genuinely given up: OpenSearch can pre-filter a kNN query by category
inside the vector search, whereas here a filtered dense query is a SQL ``WHERE``
that HNSW applies after the fact. At this corpus size that is not a real cost,
and it is written down here so it is a known trade rather than a discovered one.

**The index is approximate, and that is the point.** HNSW gives up some recall
for speed. How much is exactly the kind of claim this project refuses to
assume - Phase 3 measures it against exact search rather than trusting it.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.db.models import QuerySet
from pgvector.django import CosineDistance

from papers.models import EMBEDDING_DIMENSIONS, Chunk

logger = logging.getLogger(__name__)


class VectorError(RuntimeError):
    """Raised when a vector does not match the column it is being written to."""


def check_dimensions(vectors: list[list[float]]) -> None:
    """Reject vectors the column cannot hold, before touching the database.

    PostgreSQL would reject them too, but from inside a bulk update, with an
    error naming the column rather than the model that produced the wrong
    shape. Checking here means the message says which is which.
    """
    configured = int(settings.EMBEDDING_DIM)
    if configured != EMBEDDING_DIMENSIONS:
        raise VectorError(
            f"EMBEDDING_DIM is {configured} but the embedding column is "
            f"{EMBEDDING_DIMENSIONS}-wide. The column width is fixed by a "
            f"migration, so changing the model means a migration and a full "
            f"re-embed, not just a setting."
        )
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise VectorError(
                f"Embedder returned a {len(vector)}-dimensional vector but the "
                f"column holds {EMBEDDING_DIMENSIONS}."
            )


def store_vectors(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Write vectors onto their chunks in one statement."""
    if not chunks:
        return 0
    check_dimensions(vectors)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
    Chunk.objects.bulk_update(chunks, ["embedding"])
    return len(chunks)


def knn_search(
    vector: list[float],
    *,
    size: int = 10,
    queryset: QuerySet[Chunk] | None = None,
) -> list[tuple[Chunk, float]]:
    """Nearest chunks by cosine distance, closest first.

    Returns ``(chunk, similarity)`` where similarity is ``1 - distance``, so
    higher is better and the ordering matches every other retriever in the
    project. Mixing a distance and a score in one codebase is a reliable way to
    end up with a ranked list sorted backwards, and a backwards list still
    produces plausible-looking metrics.
    """
    base = queryset if queryset is not None else Chunk.objects.all()
    rows = (
        base.exclude(embedding__isnull=True)
        .annotate(distance=CosineDistance("embedding", vector))
        .order_by("distance")
        .select_related("paper")[:size]
    )
    return [(chunk, 1.0 - float(chunk.distance)) for chunk in rows]


def unembedded_count() -> int:
    """Chunks with no vector. Non-zero means dense retrieval has a blind spot."""
    return Chunk.objects.filter(embedding__isnull=True).count()


def stale_vector_count() -> int:
    """Chunks embedded by a model other than the configured one.

    A model change invalidates every vector the previous one produced. Mixing
    them in one index does not error - it silently returns meaningless
    neighbours, because the two models do not share a coordinate space.
    """
    return (
        Chunk.objects.exclude(embedding_model=settings.EMBEDDING_MODEL)
        .exclude(embedding__isnull=True)
        .count()
    )


__all__ = [
    "VectorError",
    "check_dimensions",
    "knn_search",
    "stale_vector_count",
    "store_vectors",
    "unembedded_count",
]
