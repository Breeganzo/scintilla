"""Fixtures and fakes for the search tests.

Nothing here touches OpenSearch or loads a transformer. The retrievers were
written to take their dependencies as constructor arguments precisely so that
their ranking logic can be tested without either, and so that a test failure
means the ranking is wrong rather than that a service was down.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from papers.models import Chunk, Paper
from search.retrievers.base import RetrievalResult


def result(arxiv_id: str, rank: int, score: float = 1.0) -> RetrievalResult:
    """A ranked result, for building expected lists by hand."""
    return RetrievalResult(arxiv_id=arxiv_id, score=score, rank=rank)


def ranked(*arxiv_ids: str) -> list[RetrievalResult]:
    """A ranked list from IDs in order, scored so that higher rank scores higher."""
    return [
        RetrievalResult(arxiv_id=arxiv_id, score=float(len(arxiv_ids) - index), rank=index + 1)
        for index, arxiv_id in enumerate(arxiv_ids)
    ]


class FakeOpenSearchClient:
    """Returns a canned hit list and records what it was asked.

    Recording the request body is the point: the title boost and the query type
    are ranking decisions, and a test that only checked the returned order would
    still pass if the boost silently disappeared.
    """

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.last_body: dict[str, Any] | None = None
        self.last_index: str | None = None

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.last_index = index
        self.last_body = body
        return {"hits": {"hits": self.hits[: body.get("size", len(self.hits))]}}


class FakeSearchIndex:
    """Stands in for ``ingestion.indexing.SearchIndex``."""

    def __init__(self, hits: list[dict[str, Any]] | None = None, alias: str = "papers") -> None:
        self.client = FakeOpenSearchClient(hits)
        self.alias = alias


def hit(arxiv_id: str, score: float, chunk_index: int = 0) -> dict[str, Any]:
    """One OpenSearch hit in the shape the real index returns."""
    return {
        "_id": f"{arxiv_id}#{chunk_index}",
        "_score": score,
        "_source": {"arxiv_id": arxiv_id, "chunk_index": chunk_index},
    }


def fake_chunk(arxiv_id: str, chunk_index: int = 0, model: str = "test-model") -> SimpleNamespace:
    """A chunk-shaped object with just the attributes the dense retriever reads."""
    return SimpleNamespace(
        chunk_index=chunk_index,
        embedding_model=model,
        paper=SimpleNamespace(arxiv_id=arxiv_id),
    )


class RecordingEmbedder:
    """Returns a fixed vector and remembers every text it was given.

    The BGE query prefix is applied inside ``embed_query`` and is invisible in
    the results, so the only way to assert it survives a refactor is to capture
    what was actually encoded.
    """

    model_name = "recording-test-embedder"

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.queries: list[str] = []
        self.documents: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[1.0] + [0.0] * (self._dimension - 1) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0] + [0.0] * (self._dimension - 1)


class StubRetriever:
    """A retriever that returns a fixed list, or raises."""

    def __init__(
        self,
        name: str,
        results: list[RetrievalResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


@pytest.fixture
def make_paper(db):
    """Create a Paper with a usable default for every required field."""

    def _make(arxiv_id: str, title: str = "A paper", abstract: str = "An abstract.") -> Paper:
        return Paper.objects.create(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=["A. Researcher"],
            categories=["hep-ex"],
            primary_category="hep-ex",
            published_at=datetime(2024, 1, 22, 12, 0, tzinfo=UTC),
            arxiv_updated_at=datetime(2024, 1, 22, 12, 0, tzinfo=UTC),
            abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        )

    return _make


@pytest.fixture
def make_chunk(db):
    """Create a Chunk on a paper, optionally with a vector."""

    def _make(paper: Paper, index: int = 0, embedding: list[float] | None = None) -> Chunk:
        return Chunk.objects.create(
            paper=paper,
            chunk_index=index,
            text=f"{paper.title}\n\n{paper.abstract}",
            token_count=32,
            embedding_model="test-model",
            embedding_dim=384,
            embedding=embedding,
        )

    return _make
