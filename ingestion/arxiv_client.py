"""Client for the arXiv Atom API.

arXiv publishes an open, unauthenticated API that returns Atom XML. The
protocol is trivial; everything difficult about this module is about being a
well-behaved client and failing predictably.

Four things this module takes seriously:

1. **Rate limiting.** arXiv asks for no more than one request every three
   seconds. That limit lives *inside* the client, not in the caller. A limit a
   caller has to remember is a limit that gets forgotten the first time
   somebody writes a quick backfill script.
2. **Timeouts on every request.** A request with no timeout can hang forever.
   Inside a scheduler that does not fail the task, it occupies the worker slot
   indefinitely and the pipeline stops without ever reporting an error.
3. **Retry only what is worth retrying.** 5xx and connection errors are
   transient, so they get exponential backoff. 4xx means the request itself is
   wrong, and repeating it just wastes arXiv's capacity.
4. **Parsing into a dataclass, never passing XML around.** XML leaking into the
   rest of the codebase would mean every consumer has to know the Atom schema,
   and swapping the source later would touch everything.

XML is parsed with ``defusedxml`` rather than ``xml.etree``. The stdlib parser
is vulnerable to entity-expansion attacks, and although arXiv is not a hostile
source today, "the remote XML we parse is trusted" is an assumption that tends
to outlive the reason it was true.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import requests
from defusedxml import ElementTree as DefusedElementTree

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# http://export.arxiv.org 301-redirects to https. Requesting https directly
# saves a round trip on every single call.

USER_AGENT = "scintilla/0.1 (+https://github.com/Breeganzo/scintilla)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# "2401.12345v2" -> ("2401.12345", 2). Also handles the pre-2007 style,
# "hep-ex/0501001v2" -> ("hep-ex/0501001", 2).
_VERSION_RE = re.compile(r"^(?P<base>.+)v(?P<version>\d+)$")

# arXiv accepts a submitted-date filter in this exact format. No separators,
# minute precision, no timezone - the API assumes UTC.
_ARXIV_DATE_FORMAT = "%Y%m%d%H%M"


class ArxivClientError(RuntimeError):
    """The API could not be queried, or answered with something unusable."""


def normalise_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends.

    arXiv wraps ``<title>`` and ``<summary>`` at roughly 80 columns, so the raw
    text is full of newlines and indentation that carry no meaning. Left alone,
    two identical abstracts that happen to be wrapped differently would produce
    different content hashes and trigger pointless re-embedding.
    """
    return " ".join(text.split())


def split_version(raw_id: str) -> tuple[str, int]:
    """Separate an arXiv identifier from its version suffix.

    The versionless identifier is the paper's stable identity - it is what
    ``Paper.arxiv_id`` stores and what makes an upsert an upsert. Keying on the
    versioned form instead would make every revision look like a brand new
    paper, and the corpus would slowly fill with duplicates of the same work.

    An identifier with no suffix is treated as version 1, which is what arXiv
    means by its absence.
    """
    match = _VERSION_RE.match(raw_id)
    if match is None:
        return raw_id, 1
    return match.group("base"), int(match.group("version"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ArxivPaper:
    """One preprint, normalised.

    Frozen because nothing downstream has any business editing a record of what
    the remote API said.
    """

    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    abs_url: str
    pdf_url: str

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"


class RateLimiter:
    """Enforces a minimum interval between calls.

    ``sleep`` and ``monotonic`` are injectable so the tests can prove the
    limiter waits the right amount without the suite actually waiting. A rate
    limiter tested by sleeping is a rate limiter nobody tests.

    ``monotonic`` rather than ``time.time`` because the wall clock can jump
    backwards - NTP corrections, daylight saving - and a negative elapsed time
    would make the limiter wait far longer than asked.
    """

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_call: float | None = None

    def wait(self) -> float:
        """Block until the minimum interval has elapsed. Returns seconds slept."""
        now = self._monotonic()

        if self._last_call is None:
            self._last_call = now
            return 0.0

        remaining = self.min_interval_seconds - (now - self._last_call)
        if remaining <= 0:
            self._last_call = now
            return 0.0

        self._sleep(remaining)
        self._last_call = self._monotonic()
        return remaining


class ArxivClient:
    """Queries arXiv and yields :class:`ArxivPaper` records."""

    def __init__(
        self,
        *,
        rate_limit_seconds: float = 3.0,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        backoff_base_seconds: float = 2.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", USER_AGENT)
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max(1, max_attempts)
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep
        self.limiter = RateLimiter(rate_limit_seconds, sleep=sleep, monotonic=monotonic)

    # ---- HTTP ------------------------------------------------------------

    def _get(self, params: dict[str, str | int]) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self.limiter.wait()

            try:
                response = self._session.get(
                    ARXIV_API_URL,
                    params=params,
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("arXiv request failed (attempt %s): %s", attempt, exc)
            else:
                if response.status_code == 200:
                    return response.text

                # 4xx other than 429 means the request is wrong. Retrying an
                # incorrect request cannot make it correct, and hammering a
                # public API with it is exactly the behaviour rate limits exist
                # to stop.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise ArxivClientError(
                        f"arXiv rejected the request with HTTP {response.status_code}"
                    )

                last_error = ArxivClientError(f"arXiv returned HTTP {response.status_code}")
                logger.warning("arXiv returned HTTP %s (attempt %s)", response.status_code, attempt)

            if attempt < self._max_attempts:
                # Exponential: 2s, 4s, 8s. Long enough for a transient upstream
                # problem to clear, short enough that a scheduled run does not
                # overrun its window.
                self._sleep(self._backoff_base_seconds * (2 ** (attempt - 1)))

        raise ArxivClientError(
            f"arXiv did not answer successfully after {self._max_attempts} attempts"
        ) from last_error

    # ---- Query construction ---------------------------------------------

    @staticmethod
    def build_search_query(
        categories: list[str],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Build the ``search_query`` term.

        ``since`` is the watermark. Without it every run re-scans the whole
        category from the beginning: correct, because the content hash still
        prevents duplicate work, but wasteful - thousands of API calls to
        discover nothing changed.
        """
        clause = " OR ".join(f"cat:{category}" for category in categories)
        query = f"({clause})"

        if since is not None:
            upper = until or datetime.now(tz=UTC)
            query += (
                f" AND submittedDate:"
                f"[{since.astimezone(UTC):{_ARXIV_DATE_FORMAT}}"
                f" TO {upper.astimezone(UTC):{_ARXIV_DATE_FORMAT}}]"
            )

        return query

    # ---- Parsing ---------------------------------------------------------

    @staticmethod
    def parse_feed(xml_text: str) -> tuple[list[ArxivPaper], int]:
        """Parse one Atom page into papers plus the reported total result count.

        Entries missing a field the corpus cannot do without are skipped with a
        warning rather than aborting the page. One malformed record should not
        cost the other ninety-nine.
        """
        try:
            root = DefusedElementTree.fromstring(xml_text)
        except Exception as exc:  # noqa: BLE001 - defusedxml raises several types
            raise ArxivClientError(f"arXiv returned unparseable XML: {exc}") from exc

        total_text = root.findtext(f"{_OPENSEARCH}totalResults")
        try:
            total = int(total_text) if total_text else 0
        except ValueError:
            total = 0

        papers: list[ArxivPaper] = []
        for entry in root.findall(f"{_ATOM}entry"):
            paper = ArxivClient._parse_entry(entry)
            if paper is not None:
                papers.append(paper)

        return papers, total

    @staticmethod
    def _parse_entry(entry) -> ArxivPaper | None:  # noqa: ANN001 - ElementTree element
        id_url = entry.findtext(f"{_ATOM}id") or ""
        raw_id = id_url.rsplit("/abs/", 1)[-1].strip()
        if not raw_id:
            logger.warning("Skipping arXiv entry with no identifier")
            return None

        arxiv_id, version = split_version(raw_id)

        title = normalise_whitespace(entry.findtext(f"{_ATOM}title") or "")
        abstract = normalise_whitespace(entry.findtext(f"{_ATOM}summary") or "")
        published = _parse_timestamp(entry.findtext(f"{_ATOM}published"))
        updated = _parse_timestamp(entry.findtext(f"{_ATOM}updated")) or published

        if not title or not abstract or published is None:
            logger.warning("Skipping arXiv entry %s: missing title, abstract or date", raw_id)
            return None

        authors = [
            normalise_whitespace(name)
            for author in entry.findall(f"{_ATOM}author")
            if (name := author.findtext(f"{_ATOM}name"))
        ]

        categories = [
            term for category in entry.findall(f"{_ATOM}category") if (term := category.get("term"))
        ]

        primary_element = entry.find(f"{_ARXIV}primary_category")
        primary_category = ""
        if primary_element is not None:
            primary_category = primary_element.get("term") or ""
        if not primary_category:
            primary_category = categories[0] if categories else ""

        abs_url = ""
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            href = link.get("href") or ""
            if link.get("title") == "pdf":
                pdf_url = href
            elif link.get("rel") == "alternate":
                abs_url = href

        versioned = f"{arxiv_id}v{version}"
        abs_url = abs_url or f"https://arxiv.org/abs/{versioned}"
        pdf_url = pdf_url or f"https://arxiv.org/pdf/{versioned}"

        return ArxivPaper(
            arxiv_id=arxiv_id,
            version=version,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            primary_category=primary_category,
            published=published,
            updated=updated or published,
            abs_url=abs_url,
            pdf_url=pdf_url,
        )

    # ---- Public API ------------------------------------------------------

    def fetch_page(
        self,
        *,
        categories: list[str],
        start: int = 0,
        page_size: int = 100,
        since: datetime | None = None,
    ) -> tuple[list[ArxivPaper], int]:
        """Fetch one page. Returns the papers and arXiv's total result count."""
        params: dict[str, str | int] = {
            "search_query": self.build_search_query(categories, since=since),
            "start": start,
            "max_results": page_size,
            # Newest first, so a run interrupted halfway has still collected the
            # papers most likely to be missing from the corpus.
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return self.parse_feed(self._get(params))

    def iter_papers(
        self,
        *,
        categories: list[str],
        limit: int,
        page_size: int = 100,
        since: datetime | None = None,
    ) -> Iterator[ArxivPaper]:
        """Yield up to ``limit`` papers, paginating as needed.

        A generator rather than a list so the pipeline can write each paper as
        it arrives. Harvesting everything into memory first would mean a
        failure on the last page discards every result before it.
        """
        yielded = 0
        start = 0

        while yielded < limit:
            batch_size = min(page_size, limit - yielded)
            papers, total = self.fetch_page(
                categories=categories,
                start=start,
                page_size=batch_size,
                since=since,
            )

            if not papers:
                # arXiv returns an empty page at the end of the result set, and
                # occasionally as a transient hiccup. Either way there is
                # nothing to yield, and continuing would loop forever.
                logger.info("arXiv returned no entries at start=%s; stopping", start)
                return

            for paper in papers:
                yield paper
                yielded += 1
                if yielded >= limit:
                    return

            start += len(papers)
            if total and start >= total:
                return
