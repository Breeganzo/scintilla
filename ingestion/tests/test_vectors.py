"""Tests for dense retrieval in PostgreSQL.

These run against a real database with the ``vector`` extension, because the
thing worth testing is that the HNSW index and the cosine operator return what
this project thinks they return. A mocked distance function would test nothing
except the mock.
"""

from __future__ import annotations

import pytest

from ingestion.embedding import HashingEmbedder
from ingestion.vectors import (
    VectorError,
    check_dimensions,
    knn_search,
    stale_vector_count,
    store_vectors,
    unembedded_count,
)
from papers.models import EMBEDDING_DIMENSIONS, Chunk, Paper

from .conftest import make_source

pytestmark = pytest.mark.django_db


def make_chunk(arxiv_id: str, text: str, *, embedding: list[float] | None = None) -> Chunk:
    source = make_source(arxiv_id, title=text)
    paper = Paper.objects.create(
        arxiv_id=source.arxiv_id,
        version=source.version,
        title=text,
        abstract=source.abstract,
        authors=source.authors,
        categories=source.categories,
        primary_category=source.primary_category,
        published_at=source.published,
        arxiv_updated_at=source.updated,
        abs_url=source.abs_url,
        pdf_url=source.pdf_url,
    )
    return Chunk.objects.create(
        paper=paper,
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
        embedding_model="hashing-test-embedder",
        embedding_dim=EMBEDDING_DIMENSIONS,
        embedding=embedding,
    )


class TestDimensionChecks:
    def test_wrong_width_is_rejected_before_the_database(self) -> None:
        """PostgreSQL would reject it too, from inside a bulk update.

        The error would name the column rather than the model that produced the
        wrong shape, which is the wrong end of the problem to be told about.
        """
        with pytest.raises(VectorError, match="384"):
            check_dimensions([[0.1] * 128])

    def test_setting_disagreeing_with_the_column_is_caught(self, settings) -> None:
        # The column width is fixed by a migration; the setting is not. If they
        # disagree, the setting is the one that is wrong.
        settings.EMBEDDING_DIM = 768
        with pytest.raises(VectorError, match="migration"):
            check_dimensions([[0.1] * EMBEDDING_DIMENSIONS])

    def test_correct_width_passes(self) -> None:
        check_dimensions([[0.1] * EMBEDDING_DIMENSIONS])


class TestStoreVectors:
    def test_vectors_are_written(self) -> None:
        chunk = make_chunk("2401.00001", "neutrino oscillation measurement")
        embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
        store_vectors([chunk], embedder.embed_documents([chunk.text]))

        chunk.refresh_from_db()
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS

    def test_empty_input_is_a_no_op(self) -> None:
        assert store_vectors([], []) == 0


class TestKnnSearch:
    def test_closest_vector_ranks_first(self) -> None:
        embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
        target = make_chunk("2401.00001", "search for long-lived particles at the collider")
        other = make_chunk("2401.00002", "medieval agricultural practice in northern europe")
        for chunk in (target, other):
            store_vectors([chunk], embedder.embed_documents([chunk.text]))

        results = knn_search(embedder.embed_query("long-lived particles collider search"))

        assert results[0][0].pk == target.pk

    def test_similarity_is_returned_not_distance(self) -> None:
        """Higher must mean better, matching every other retriever here.

        Mixing a distance and a score in one codebase is a reliable way to end
        up with a ranked list sorted backwards - and a backwards list still
        produces plausible-looking metrics.
        """
        embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
        chunk = make_chunk("2401.00001", "top quark mass measurement")
        vector = embedder.embed_documents([chunk.text])[0]
        store_vectors([chunk], [vector])

        ((_, similarity),) = knn_search(vector, size=1)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_unembedded_chunks_are_excluded(self) -> None:
        # A NULL vector is not a distant one. Including it would put it
        # somewhere arbitrary in the ranking rather than nowhere.
        embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
        embedded = make_chunk("2401.00001", "higgs boson decay channel")
        make_chunk("2401.00002", "higgs boson decay channel duplicate")
        store_vectors([embedded], embedder.embed_documents([embedded.text]))

        results = knn_search(embedder.embed_query("higgs boson"))
        assert [chunk.pk for chunk, _ in results] == [embedded.pk]

    def test_size_limits_results(self) -> None:
        embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
        for n in range(5):
            chunk = make_chunk(f"2401.0000{n}", f"detector calibration study {n}")
            store_vectors([chunk], embedder.embed_documents([chunk.text]))

        assert len(knn_search(embedder.embed_query("detector calibration"), size=2)) == 2


class TestCoverageCounters:
    def test_unembedded_count(self) -> None:
        make_chunk("2401.00001", "one", embedding=[0.0] * (EMBEDDING_DIMENSIONS - 1) + [1.0])
        make_chunk("2401.00002", "two")
        assert unembedded_count() == 1

    def test_stale_vectors_are_counted(self, settings) -> None:
        """A model change invalidates every vector the previous one produced.

        Mixing them does not error - the two models do not share a coordinate
        space, so the neighbours are simply meaningless.
        """
        make_chunk("2401.00001", "one", embedding=[0.0] * (EMBEDDING_DIMENSIONS - 1) + [1.0])
        settings.EMBEDDING_MODEL = "some/other-model"
        assert stale_vector_count() == 1

    def test_matching_model_is_not_stale(self, settings) -> None:
        make_chunk("2401.00001", "one", embedding=[0.0] * (EMBEDDING_DIMENSIONS - 1) + [1.0])
        settings.EMBEDDING_MODEL = "hashing-test-embedder"
        assert stale_vector_count() == 0
