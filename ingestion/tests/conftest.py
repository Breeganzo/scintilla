"""Shared fixtures for the ingestion suite.

Every test here runs offline. The arXiv responses are real, recorded once and
replayed from disk.

**Why no network in the test suite.** A test that calls arXiv fails when arXiv
is slow, fails when the runner has no egress, and takes three seconds per case
because of the rate limiter. A suite that fails for reasons unrelated to the
code is a suite people learn to ignore, and an ignored suite is worse than no
suite - it produces false confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from opensearchpy.exceptions import NotFoundError

from ingestion.arxiv_client import ArxivClient, ArxivPaper
from ingestion.embedding import HashingEmbedder
from ingestion.indexing import SearchIndex
from papers.models import EMBEDDING_DIMENSIONS

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def arxiv_page_xml() -> str:
    """A real five-entry hep-ex response, recorded from the live API."""
    return load_fixture("arxiv_page.xml")


@pytest.fixture
def arxiv_empty_xml() -> str:
    """A real response for a query that matched nothing."""
    return load_fixture("arxiv_empty.xml")


@dataclass
class FakeResponse:
    status_code: int
    text: str = ""


class FakeSession:
    """Stands in for ``requests.Session``.

    Records every call so tests can assert on the parameters actually sent -
    the rate limit and the date window are only useful if they reach the wire.
    """

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []

    def get(self, url, *, params=None, timeout=None):  # noqa: ANN001
        self.calls.append({"url": url, "params": params or {}, "timeout": timeout})
        if not self._responses:
            return FakeResponse(status_code=200, text="")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """Monotonic clock and sleep that advance a counter instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    @property
    def total_slept(self) -> float:
        return sum(self.slept)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def build_client(
    responses: list[FakeResponse | Exception],
    clock: FakeClock,
    **kwargs,
) -> tuple[ArxivClient, FakeSession]:
    session = FakeSession(responses)
    client = ArxivClient(
        session=session,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )
    return client, session


def make_source(
    arxiv_id: str = "2401.00001",
    *,
    version: int = 1,
    title: str = "Search for long-lived particles in proton-proton collisions",
    abstract: str = (
        "We report a search for long-lived particles decaying in the outer "
        "detector. No significant excess above the Standard Model expectation "
        "is observed and limits are set at 95% confidence level."
    ),
) -> ArxivPaper:
    """An :class:`ArxivPaper` as the client would have produced it."""
    published = datetime(2024, 1, 15, 9, 0, tzinfo=UTC)
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=["A. Researcher", "B. Collaborator"],
        categories=["hep-ex", "hep-ph"],
        primary_category="hep-ex",
        published=published,
        updated=published,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}v{version}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v{version}",
    )


class StubClient:
    """Yields a fixed list of papers. Lets the pipeline be tested with no HTTP."""

    def __init__(self, papers: list[ArxivPaper]) -> None:
        self.papers = papers
        self.calls: list[dict] = []

    def iter_papers(self, *, categories, limit, page_size=100, since=None):  # noqa: ANN001
        self.calls.append(
            {"categories": categories, "limit": limit, "page_size": page_size, "since": since}
        )
        yield from self.papers[:limit]


# ---------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------
#
# An in-memory stand-in rather than a live node. The indexing logic worth
# testing is ours - per-item bulk error handling, alias switching, what gets
# marked as indexed and when - and none of it needs a JVM. Tests that require a
# running cluster get skipped on the machines where it matters most, so the
# behaviour they cover ends up effectively untested.
#
# The mapping itself is verified against a real node in the Day 5 manual
# checks, because only a real node can reject it.


class FakeIndices:
    """The ``client.indices`` namespace."""

    def __init__(self, store: FakeOpenSearch) -> None:
        self.store = store

    def exists(self, index: str) -> bool:
        return index in self.store.documents

    def create(self, index: str, body: dict | None = None) -> dict:
        self.store.documents.setdefault(index, {})
        self.store.bodies[index] = body or {}
        self.store.created.append(index)
        return {"acknowledged": True}

    def get_alias(self, name: str) -> dict:
        matched = {
            index: {"aliases": {name: {}}}
            for index, aliases in self.store.aliases.items()
            if name in aliases
        }
        if not matched:
            raise NotFoundError(404, "alias_not_found_exception", {"alias": name})
        return matched

    def update_aliases(self, body: dict) -> dict:
        # Recorded whole so a test can assert the remove and the add travelled
        # in one request. Two requests would leave a window with no alias.
        self.store.alias_calls.append(body)
        for action in body["actions"]:
            if "remove" in action:
                spec = action["remove"]
                self.store.aliases.get(spec["index"], set()).discard(spec["alias"])
            if "add" in action:
                spec = action["add"]
                self.store.aliases.setdefault(spec["index"], set()).add(spec["alias"])
        return {"acknowledged": True}

    def refresh(self, index: str | None = None) -> dict:
        self.store.refreshes.append(index)
        return {"_shards": {"failed": 0}}


class FakeOpenSearch:
    """In-memory OpenSearch with just enough behaviour to be honest.

    ``fail_ids`` makes named documents fail inside a bulk request the way a
    real one does: HTTP 200 overall, the error buried in the per-item response.
    ``retry_fail_ids`` makes the individual retry fail too, which is the only
    path that should ever produce a reported failure.
    """

    def __init__(
        self,
        *,
        fail_ids: set[str] | None = None,
        retry_fail_ids: set[str] | None = None,
    ) -> None:
        self.documents: dict[str, dict[str, dict]] = {}
        self.bodies: dict[str, dict] = {}
        self.aliases: dict[str, set[str]] = {}
        self.created: list[str] = []
        self.alias_calls: list[dict] = []
        self.refreshes: list[str | None] = []
        self.bulk_calls: list[list] = []
        self.deleted_by_query: list[dict] = []
        self.fail_ids = fail_ids or set()
        self.retry_fail_ids = retry_fail_ids or set()
        self.indices = FakeIndices(self)

    def _resolve(self, index: str) -> str:
        for name, aliases in self.aliases.items():
            if index in aliases:
                return name
        return index

    def bulk(self, body: list) -> dict:
        self.bulk_calls.append(body)
        items = []
        errors = False
        for action, source in zip(body[0::2], body[1::2], strict=True):
            spec = action["index"]
            index, doc_id = spec["_index"], spec["_id"]
            if doc_id in self.fail_ids:
                errors = True
                items.append(
                    {
                        "index": {
                            "_id": doc_id,
                            "status": 400,
                            "error": {
                                "type": "mapper_parsing_exception",
                                "reason": "simulated per-item failure",
                            },
                        }
                    }
                )
                continue
            self.documents.setdefault(index, {})[doc_id] = source
            items.append({"index": {"_id": doc_id, "status": 201}})
        return {"errors": errors, "items": items}

    def index(self, index: str, id: str, body: dict) -> dict:  # noqa: A002
        if id in self.retry_fail_ids:
            raise RuntimeError("simulated retry failure")
        self.documents.setdefault(index, {})[id] = body
        return {"result": "created"}

    def count(self, index: str) -> dict:
        return {"count": len(self.documents.get(self._resolve(index), {}))}

    def delete_by_query(self, index: str, body: dict, conflicts: str = "abort") -> dict:
        self.deleted_by_query.append({"index": index, "body": body})
        filters = body["query"]["bool"]["filter"]
        wanted = set(filters[0]["terms"]["arxiv_id"])
        minimum = filters[1]["range"]["chunk_index"]["gte"]
        target = self.documents.setdefault(self._resolve(index), {})
        doomed = [
            doc_id
            for doc_id, source in target.items()
            if source["arxiv_id"] in wanted and source["chunk_index"] >= minimum
        ]
        for doc_id in doomed:
            del target[doc_id]
        return {"deleted": len(doomed)}

    def search(self, index: str, body: dict) -> dict:
        sources = self.documents.get(self._resolve(index), {})
        hits = [
            {"_id": doc_id, "_score": 1.0, "_source": source}
            for doc_id, source in list(sources.items())[: body.get("size", 10)]
        ]
        return {"hits": {"total": {"value": len(sources)}, "hits": hits}}


@pytest.fixture
def fake_opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def search_index(fake_opensearch: FakeOpenSearch) -> SearchIndex:
    return SearchIndex(fake_opensearch, alias="papers-test")


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Matches the width of the pgvector column, which is fixed by a migration."""
    return HashingEmbedder(dimension=EMBEDDING_DIMENSIONS)
