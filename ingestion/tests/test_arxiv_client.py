"""Tests for the arXiv client. No network, no real sleeping."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import requests

from ingestion.arxiv_client import (
    ArxivClient,
    ArxivClientError,
    RateLimiter,
    normalise_whitespace,
    split_version,
)
from ingestion.tests.conftest import FakeResponse, build_client


class TestSplitVersion:
    def test_strips_a_version_suffix(self):
        assert split_version("2401.12345v2") == ("2401.12345", 2)

    def test_absent_suffix_means_version_one(self):
        assert split_version("2401.12345") == ("2401.12345", 1)

    def test_handles_the_pre_2007_identifier_style(self):
        # These still appear in hep-ex and would be mangled by a naive
        # split on "v" or on "/".
        assert split_version("hep-ex/0501001v3") == ("hep-ex/0501001", 3)

    def test_double_digit_versions(self):
        assert split_version("2401.12345v12") == ("2401.12345", 12)


class TestNormaliseWhitespace:
    def test_collapses_newlines_and_indentation(self):
        raw = "Measurement of the\n  Higgs boson   coupling\n"
        assert normalise_whitespace(raw) == "Measurement of the Higgs boson coupling"

    def test_differently_wrapped_text_normalises_identically(self):
        # This is the property that stops a re-wrapped abstract from reading as
        # a content change and triggering a pointless re-embed.
        assert normalise_whitespace("a b\nc") == normalise_whitespace("a\n  b   c")


class TestRateLimiter:
    def test_first_call_does_not_wait(self, clock):
        limiter = RateLimiter(3.0, sleep=clock.sleep, monotonic=clock.monotonic)
        assert limiter.wait() == 0.0
        assert clock.slept == []

    def test_second_immediate_call_waits_the_full_interval(self, clock):
        limiter = RateLimiter(3.0, sleep=clock.sleep, monotonic=clock.monotonic)
        limiter.wait()
        assert limiter.wait() == pytest.approx(3.0)
        assert clock.total_slept == pytest.approx(3.0)

    def test_no_wait_when_the_interval_has_already_passed(self, clock):
        limiter = RateLimiter(3.0, sleep=clock.sleep, monotonic=clock.monotonic)
        limiter.wait()
        clock.now += 10.0
        assert limiter.wait() == 0.0
        assert clock.slept == []

    def test_ten_calls_take_at_least_nine_intervals(self, clock):
        limiter = RateLimiter(3.0, sleep=clock.sleep, monotonic=clock.monotonic)
        for _ in range(10):
            limiter.wait()
        assert clock.total_slept == pytest.approx(27.0)


class TestParsing:
    def test_parses_the_recorded_page(self, arxiv_page_xml):
        papers, total = ArxivClient.parse_feed(arxiv_page_xml)
        assert len(papers) == 5
        assert total > 0

    def test_every_parsed_paper_has_the_fields_the_corpus_requires(self, arxiv_page_xml):
        papers, _ = ArxivClient.parse_feed(arxiv_page_xml)
        for paper in papers:
            assert paper.arxiv_id
            assert paper.version >= 1
            assert paper.title
            assert paper.abstract
            assert paper.primary_category
            assert paper.published.tzinfo is not None
            assert paper.abs_url.startswith("http")
            assert paper.pdf_url.startswith("http")

    def test_identifiers_are_stored_without_the_version_suffix(self, arxiv_page_xml):
        papers, _ = ArxivClient.parse_feed(arxiv_page_xml)
        assert all(not paper.arxiv_id.endswith(f"v{paper.version}") for paper in papers)
        assert all(paper.versioned_id.endswith(f"v{paper.version}") for paper in papers)

    def test_titles_and_abstracts_are_whitespace_normalised(self, arxiv_page_xml):
        papers, _ = ArxivClient.parse_feed(arxiv_page_xml)
        for paper in papers:
            assert "\n" not in paper.title
            assert "\n" not in paper.abstract
            assert "  " not in paper.abstract

    def test_empty_result_set_parses_to_no_papers(self, arxiv_empty_xml):
        papers, total = ArxivClient.parse_feed(arxiv_empty_xml)
        assert papers == []
        assert total == 0

    def test_unparseable_xml_raises_rather_than_returning_nothing(self):
        # Returning [] here would look identical to "arXiv has no new papers",
        # and the pipeline would record a clean run while silently losing data.
        with pytest.raises(ArxivClientError):
            ArxivClient.parse_feed("<feed><entry>truncated")

    def test_entry_missing_an_abstract_is_skipped_not_fatal(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <title>Good paper</title>
            <summary>A real abstract.</summary>
            <published>2024-01-01T00:00:00Z</published>
            <updated>2024-01-01T00:00:00Z</updated>
          </entry>
          <entry>
            <id>http://arxiv.org/abs/2401.00002v1</id>
            <title>Broken paper</title>
            <published>2024-01-01T00:00:00Z</published>
          </entry>
        </feed>"""
        papers, _ = ArxivClient.parse_feed(xml)
        assert [p.arxiv_id for p in papers] == ["2401.00001"]

    def test_primary_category_falls_back_to_the_first_category(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00003v1</id>
            <title>No primary category</title>
            <summary>Abstract text.</summary>
            <published>2024-01-01T00:00:00Z</published>
            <updated>2024-01-01T00:00:00Z</updated>
            <category term="hep-ex"/>
          </entry>
        </feed>"""
        papers, _ = ArxivClient.parse_feed(xml)
        assert papers[0].primary_category == "hep-ex"


class TestSearchQuery:
    def test_multiple_categories_are_or_ed(self):
        query = ArxivClient.build_search_query(["hep-ex", "hep-th"])
        assert query == "(cat:hep-ex OR cat:hep-th)"

    def test_watermark_adds_a_submitted_date_window(self):
        query = ArxivClient.build_search_query(
            ["hep-ex"],
            since=datetime(2024, 1, 2, 3, 4, tzinfo=UTC),
            until=datetime(2024, 6, 7, 8, 9, tzinfo=UTC),
        )
        assert "submittedDate:[202401020304 TO 202406070809]" in query


class TestHttpBehaviour:
    def test_successful_request_sends_a_timeout_and_user_agent(self, clock, arxiv_page_xml):
        client, session = build_client(
            [FakeResponse(200, arxiv_page_xml)], clock, rate_limit_seconds=3.0
        )
        papers, _ = client.fetch_page(categories=["hep-ex"])

        assert len(papers) == 5
        assert session.calls[0]["timeout"] == 30.0
        assert "scintilla" in session.headers["User-Agent"]

    def test_rate_limit_is_applied_between_pages(self, clock, arxiv_page_xml):
        client, _ = build_client(
            [FakeResponse(200, arxiv_page_xml), FakeResponse(200, arxiv_page_xml)],
            clock,
            rate_limit_seconds=3.0,
        )
        client.fetch_page(categories=["hep-ex"])
        client.fetch_page(categories=["hep-ex"], start=5)

        assert clock.total_slept == pytest.approx(3.0)

    def test_retries_on_503_then_succeeds(self, clock, arxiv_page_xml):
        client, session = build_client(
            [FakeResponse(503), FakeResponse(200, arxiv_page_xml)],
            clock,
            rate_limit_seconds=0.0,
            backoff_base_seconds=2.0,
        )
        papers, _ = client.fetch_page(categories=["hep-ex"])

        assert len(papers) == 5
        assert len(session.calls) == 2
        assert clock.slept == [2.0]

    def test_retries_on_a_connection_error(self, clock, arxiv_page_xml):
        client, session = build_client(
            [requests.ConnectionError("reset"), FakeResponse(200, arxiv_page_xml)],
            clock,
            rate_limit_seconds=0.0,
        )
        papers, _ = client.fetch_page(categories=["hep-ex"])
        assert len(papers) == 5
        assert len(session.calls) == 2

    def test_backoff_is_exponential_and_gives_up(self, clock):
        client, session = build_client(
            [FakeResponse(503)] * 4,
            clock,
            rate_limit_seconds=0.0,
            max_attempts=4,
            backoff_base_seconds=2.0,
        )
        with pytest.raises(ArxivClientError):
            client.fetch_page(categories=["hep-ex"])

        assert len(session.calls) == 4
        assert clock.slept == [2.0, 4.0, 8.0]

    def test_client_error_is_not_retried(self, clock):
        # Repeating a malformed request cannot make it valid, and hammering a
        # public API with one is the behaviour rate limits exist to prevent.
        client, session = build_client([FakeResponse(400)], clock, rate_limit_seconds=0.0)
        with pytest.raises(ArxivClientError):
            client.fetch_page(categories=["hep-ex"])
        assert len(session.calls) == 1


class TestPagination:
    def test_stops_at_the_requested_limit(self, clock, arxiv_page_xml):
        client, _ = build_client(
            [FakeResponse(200, arxiv_page_xml)] * 5, clock, rate_limit_seconds=0.0
        )
        papers = list(client.iter_papers(categories=["hep-ex"], limit=3, page_size=5))
        assert len(papers) == 3

    def test_an_empty_page_ends_iteration_rather_than_looping(self, clock, arxiv_empty_xml):
        client, session = build_client(
            [FakeResponse(200, arxiv_empty_xml)] * 3, clock, rate_limit_seconds=0.0
        )
        papers = list(client.iter_papers(categories=["hep-ex"], limit=100, page_size=5))
        assert papers == []
        assert len(session.calls) == 1
