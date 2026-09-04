"""Tests for the ingestion pipeline, and above all for its idempotency.

The phase gate for this work is: run it twice, and the second run must do
nothing. These tests are that gate, expressed as assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ingestion.pipeline import ingest, resolve_watermark
from ingestion.tests.conftest import StubClient, make_source
from papers.models import Chunk, IngestionRun, Paper

pytestmark = pytest.mark.django_db


def run(papers, **kwargs):
    return ingest(
        categories=kwargs.pop("categories", ["hep-ex"]),
        limit=kwargs.pop("limit", 100),
        client=StubClient(papers),
        **kwargs,
    )


class TestFirstRun:
    def test_creates_papers(self):
        result = run([make_source("2401.00001"), make_source("2401.00002")])

        assert result.created == 2
        assert result.updated == 0
        assert result.skipped == 0
        assert Paper.objects.count() == 2

    def test_stores_the_versionless_identifier(self):
        run([make_source("2401.00001", version=3)])
        paper = Paper.objects.get()

        assert paper.arxiv_id == "2401.00001"
        assert paper.version == 3

    def test_creates_one_chunk_per_paper(self):
        run([make_source("2401.00001")])
        assert Chunk.objects.count() == 1
        assert Chunk.objects.get().chunk_index == 0

    def test_the_chunk_contains_the_title(self):
        source = make_source("2401.00001")
        run([source])
        assert source.title in Chunk.objects.get().text

    def test_new_papers_are_not_yet_indexed(self):
        # Postgres and the search index are allowed to disagree. This field is
        # what makes the disagreement visible instead of invisible.
        run([make_source("2401.00001")])
        paper = Paper.objects.get()

        assert paper.indexed_at is None
        assert paper.needs_indexing is True

    def test_writes_a_successful_ledger_row(self):
        result = run([make_source("2401.00001")])
        ledger = IngestionRun.objects.get(pk=result.run_id)

        assert ledger.status == IngestionRun.Status.SUCCESS
        assert ledger.papers_seen == 1
        assert ledger.papers_created == 1
        assert ledger.finished_at is not None
        assert ledger.duration_seconds is not None
        assert ledger.categories == ["hep-ex"]


class TestSecondRunIsANoOp:
    """The phase gate. Re-running must be free and must change nothing."""

    def test_second_run_creates_nothing(self):
        papers = [make_source("2401.00001"), make_source("2401.00002")]
        run(papers)
        second = run(papers, use_watermark=False)

        assert second.created == 0
        assert second.updated == 0
        assert second.changed == 0

    def test_second_run_skips_every_paper(self):
        papers = [make_source("2401.00001"), make_source("2401.00002")]
        run(papers)
        second = run(papers, use_watermark=False)

        assert second.seen == 2
        assert second.skipped == 2

    def test_second_run_does_not_duplicate_rows(self):
        papers = [make_source("2401.00001"), make_source("2401.00002")]
        run(papers)
        run(papers, use_watermark=False)

        assert Paper.objects.count() == 2
        assert Chunk.objects.count() == 2

    def test_second_run_does_not_touch_the_rows(self):
        # The skip happens before any write, so updated_at must not move. If it
        # did, every paper would look stale to the re-indexing query and the
        # saving would be lost.
        run([make_source("2401.00001")])
        before = Paper.objects.get().updated_at

        run([make_source("2401.00001")], use_watermark=False)
        assert Paper.objects.get().updated_at == before

    def test_ten_runs_are_still_one_paper(self):
        for _ in range(10):
            run([make_source("2401.00001")], use_watermark=False)
        assert Paper.objects.count() == 1


class TestRevisions:
    def test_changed_abstract_updates_in_place(self):
        run([make_source("2401.00001")])
        revised = make_source("2401.00001", version=2, abstract="A completely rewritten abstract.")
        result = run([revised], use_watermark=False)

        assert result.updated == 1
        assert result.created == 0
        assert Paper.objects.count() == 1

        paper = Paper.objects.get()
        assert paper.abstract == "A completely rewritten abstract."
        assert paper.version == 2

    def test_a_revision_marks_the_paper_for_re_indexing(self):
        run([make_source("2401.00001")])
        Paper.objects.update(indexed_at=datetime(2024, 6, 1, tzinfo=UTC))

        run(
            [make_source("2401.00001", abstract="Different text entirely.")],
            use_watermark=False,
        )
        assert Paper.objects.get().indexed_at is None

    def test_a_revision_replaces_the_chunks(self):
        run([make_source("2401.00001")])
        run(
            [make_source("2401.00001", abstract="Different text entirely.")],
            use_watermark=False,
        )

        assert Chunk.objects.count() == 1
        assert "Different text entirely." in Chunk.objects.get().text

    def test_a_version_bump_with_identical_text_is_skipped(self):
        # A new version number with unchanged title and abstract cannot change a
        # search result, so it is deliberately not worth a write.
        run([make_source("2401.00001", version=1)])
        result = run([make_source("2401.00001", version=2)], use_watermark=False)

        assert result.skipped == 1
        assert result.updated == 0

    def test_a_title_change_alone_counts_as_a_revision(self):
        run([make_source("2401.00001")])
        result = run(
            [make_source("2401.00001", title="An entirely different title")],
            use_watermark=False,
        )
        assert result.updated == 1


class TestWatermark:
    def test_a_successful_run_records_the_newest_publication_date(self):
        result = run([make_source("2401.00001")])
        ledger = IngestionRun.objects.get(pk=result.run_id)

        assert ledger.watermark == datetime(2024, 1, 15, 9, 0, tzinfo=UTC)

    def test_the_next_run_resumes_from_it(self):
        run([make_source("2401.00001")])
        client = StubClient([])
        ingest(categories=["hep-ex"], limit=10, client=client)

        assert client.calls[0]["since"] == datetime(2024, 1, 15, 9, 0, tzinfo=UTC)

    def test_full_scan_ignores_the_watermark(self):
        run([make_source("2401.00001")])
        client = StubClient([])
        ingest(categories=["hep-ex"], limit=10, client=client, use_watermark=False)

        assert client.calls[0]["since"] is None

    def test_watermarks_do_not_leak_between_category_sets(self):
        # Having harvested hep-ex up to today says nothing about how far hep-th
        # has been harvested. Reusing the watermark would skip everything before it.
        run([make_source("2401.00001")], categories=["hep-ex"])
        assert resolve_watermark(["hep-th"]) is None

    def test_a_failed_run_does_not_advance_the_watermark(self):
        class ExplodingClient:
            def iter_papers(self, **kwargs):  # noqa: ANN003
                raise RuntimeError("arXiv unreachable")
                yield  # pragma: no cover

        with pytest.raises(RuntimeError):
            ingest(categories=["hep-ex"], limit=10, client=ExplodingClient())

        ledger = IngestionRun.objects.latest("started_at")
        assert ledger.status == IngestionRun.Status.FAILED
        assert ledger.watermark is None


class TestFailureHandling:
    def test_one_bad_record_does_not_lose_the_rest(self):
        # A single malformed paper costing the other ninety-nine would make the
        # whole run worthless.
        bad = make_source("x" * 64)  # exceeds arxiv_id max_length
        result = run([make_source("2401.00001"), bad, make_source("2401.00002")])

        assert result.created == 2
        assert result.errors
        assert result.status == IngestionRun.Status.PARTIAL

    def test_partial_status_is_recorded_on_the_ledger(self):
        bad = make_source("x" * 64)
        result = run([make_source("2401.00001"), bad])
        ledger = IngestionRun.objects.get(pk=result.run_id)

        assert ledger.status == IngestionRun.Status.PARTIAL
        assert ledger.error_message

    def test_a_failed_run_still_leaves_a_ledger_row(self):
        class ExplodingClient:
            def iter_papers(self, **kwargs):  # noqa: ANN003
                raise RuntimeError("boom")
                yield  # pragma: no cover

        with pytest.raises(RuntimeError):
            ingest(categories=["hep-ex"], limit=10, client=ExplodingClient())

        assert IngestionRun.objects.filter(status=IngestionRun.Status.FAILED).count() == 1


class TestManagementCommand:
    def test_command_runs_and_reports(self, monkeypatch, capsys):
        from django.core.management import call_command

        import ingestion.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module.ArxivClient,
            "iter_papers",
            lambda self, **kwargs: iter([make_source("2401.00001")]),
        )
        call_command("ingest_arxiv", "--categories", "hep-ex", "--limit", "1")

        output = capsys.readouterr().out
        assert "status=success" in output
        assert "created=1" in output
        assert Paper.objects.count() == 1
