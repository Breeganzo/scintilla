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

from ingestion.arxiv_client import ArxivClient, ArxivPaper

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
