"""The ingestion pipeline: arXiv in, PostgreSQL out, safe to re-run.

**Why idempotency is the central concern here**

An ingestion job is not a one-off script. It runs on a schedule, it will fail
partway through at some point, and somebody will re-run it by hand while
debugging. If a re-run duplicates records or silently redoes expensive work,
none of that is safe and the corpus stops being trustworthy - which makes every
retrieval number measured against it meaningless.

Three independent mechanisms, so a failure at any stage is recoverable:

1. **Content hash.** ``sha256(normalised title + abstract)``, compared *before*
   any work. Unchanged hash means the paper is skipped without embedding it.
   Embedding is the expensive step, so this is where the saving is.
2. **Upsert on the versionless ``arxiv_id``.** Papers get revised. Keying on the
   stable identifier means a revision updates the existing row instead of
   inserting a near-duplicate.
3. **Deterministic search-index document ID** (Day 5). Re-indexing overwrites
   rather than appending, so the index cannot drift from the database.

Plus a **watermark**: each successful run records the newest ``published`` date
it saw, and the next run asks arXiv only for papers after it. Without this,
every run re-scans the entire category from the beginning - still correct,
because the hash catches the duplicates, but thousands of pointless API calls.

The watermark has an obvious hole, and mechanism 1 is what plugs it: a paper
revised *today* but originally published *before* the watermark would never be
re-fetched by a date query. So the watermark is treated as an optimisation, not
as the source of truth, and a periodic full re-scan stays cheap precisely
because unchanged hashes cost nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ingestion.arxiv_client import ArxivClient, ArxivPaper
from ingestion.chunking import TokenCounter, chunk_paper
from papers.models import Chunk, IngestionRun, Paper

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """What one run did. Mirrors the IngestionRun row, for callers that want it
    without a database round trip."""

    run_id: int
    status: str
    seen: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    chunks_written: int = 0
    papers_split: int = 0
    watermark: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        """Papers that actually required work. Zero on a clean re-run."""
        return self.created + self.updated


def resolve_watermark(categories: list[str]) -> datetime | None:
    """Newest publication date the last successful run saw for these categories.

    Matched on the exact category set: harvesting ``hep-ex`` up to today says
    nothing about how far ``hep-th`` has been harvested, and reusing one
    category's watermark for another would silently skip everything before it.
    """
    last_success = (
        IngestionRun.objects.filter(
            status=IngestionRun.Status.SUCCESS,
            categories=sorted(categories),
            watermark__isnull=False,
        )
        .order_by("-started_at")
        .first()
    )
    return last_success.watermark if last_success else None


def _chunks_for(
    paper: Paper, source: ArxivPaper, count_tokens: TokenCounter | None = None
) -> tuple[list[Chunk], bool]:
    pieces = chunk_paper(
        source.title,
        source.abstract,
        arxiv_id=source.arxiv_id,
        count_tokens=count_tokens,
    )
    chunks = [
        Chunk(
            paper=paper,
            chunk_index=piece.index,
            text=piece.text,
            token_count=piece.token_count,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dim=settings.EMBEDDING_DIM,
            # NULL until Day 5 writes it to the search index. The two stores are
            # allowed to disagree; this field is how that disagreement is
            # visible rather than invisible.
            indexed_at=None,
        )
        for piece in pieces
    ]
    return chunks, len(pieces) > 1


@transaction.atomic
def _create_paper(source: ArxivPaper, count_tokens: TokenCounter | None = None) -> tuple[int, bool]:
    paper = Paper.objects.create(
        arxiv_id=source.arxiv_id,
        version=source.version,
        title=source.title,
        abstract=source.abstract,
        authors=source.authors,
        categories=source.categories,
        primary_category=source.primary_category,
        published_at=source.published,
        arxiv_updated_at=source.updated,
        abs_url=source.abs_url,
        pdf_url=source.pdf_url,
    )
    chunks, was_split = _chunks_for(paper, source, count_tokens)
    Chunk.objects.bulk_create(chunks)
    return len(chunks), was_split


@transaction.atomic
def _update_paper(
    paper: Paper, source: ArxivPaper, count_tokens: TokenCounter | None = None
) -> tuple[int, bool]:
    paper.version = source.version
    paper.title = source.title
    paper.abstract = source.abstract
    paper.authors = source.authors
    paper.categories = source.categories
    paper.primary_category = source.primary_category
    paper.published_at = source.published
    paper.arxiv_updated_at = source.updated
    paper.abs_url = source.abs_url
    paper.pdf_url = source.pdf_url
    # The content changed, so whatever is in the search index is now wrong.
    # Clearing this is what makes the paper visible to the re-indexing query.
    paper.indexed_at = None
    paper.save()

    # Replace rather than reconcile. Chunk boundaries can move when the text
    # changes, so matching old chunks to new ones is guesswork; deleting inside
    # the same transaction is exact and cheap at one chunk per paper.
    paper.chunks.all().delete()
    chunks, was_split = _chunks_for(paper, source, count_tokens)
    Chunk.objects.bulk_create(chunks)
    return len(chunks), was_split


def ingest(
    *,
    categories: list[str],
    limit: int,
    page_size: int = 100,
    client: ArxivClient | None = None,
    triggered_by: str = "manual",
    since: datetime | None = None,
    use_watermark: bool = True,
    count_tokens: TokenCounter | None = None,
) -> IngestionResult:
    """Harvest ``categories`` from arXiv into PostgreSQL.

    Every run writes an :class:`~papers.models.IngestionRun` row, including
    failed ones. A run that leaves no trace is a run nobody can debug.

    ``count_tokens`` is the embedding model's tokenizer. Passing it makes the
    chunk-size ceiling exact; leaving it out falls back to a whitespace
    heuristic that under-counts LaTeX-heavy abstracts.
    """
    categories = sorted(categories)
    client = client or ArxivClient(
        rate_limit_seconds=settings.ARXIV_RATE_LIMIT_SECONDS,
    )

    if since is None and use_watermark:
        since = resolve_watermark(categories)
        if since is not None:
            logger.info("Resuming from watermark %s", since.isoformat())

    run = IngestionRun.objects.create(
        categories=categories,
        triggered_by=triggered_by,
        status=IngestionRun.Status.RUNNING,
    )
    result = IngestionResult(run_id=run.pk, status=IngestionRun.Status.RUNNING)

    try:
        for source in client.iter_papers(
            categories=categories, limit=limit, page_size=page_size, since=since
        ):
            result.seen += 1

            if result.watermark is None or source.published > result.watermark:
                result.watermark = source.published

            try:
                existing = Paper.objects.filter(arxiv_id=source.arxiv_id).first()
                incoming_hash = Paper.compute_content_hash(source.title, source.abstract)

                if existing is None:
                    written, was_split = _create_paper(source, count_tokens)
                    result.created += 1
                elif existing.content_hash == incoming_hash:
                    # Mechanism 1. Nothing that affects retrieval has changed,
                    # so nothing is written - not even a metadata-only version
                    # bump. A new version number with identical text cannot
                    # change a search result, and writing it would cost a row
                    # update on every paper on every run for no benefit.
                    result.skipped += 1
                    continue
                else:
                    written, was_split = _update_paper(existing, source, count_tokens)
                    result.updated += 1

                result.chunks_written += written
                result.papers_split += int(was_split)

            except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
                message = f"{source.arxiv_id}: {exc}"
                result.errors.append(message)
                logger.exception("Failed to ingest %s", source.arxiv_id)

        result.status = (
            IngestionRun.Status.PARTIAL if result.errors else IngestionRun.Status.SUCCESS
        )

    except Exception as exc:  # noqa: BLE001 - recorded on the ledger, then re-raised
        result.status = IngestionRun.Status.FAILED
        result.errors.append(str(exc))
        _finalise(run, result)
        logger.exception("Ingestion run %s failed", run.pk)
        raise

    _finalise(run, result)
    logger.info(
        "Run %s %s: seen=%s created=%s updated=%s skipped=%s chunks=%s",
        run.pk,
        result.status,
        result.seen,
        result.created,
        result.updated,
        result.skipped,
        result.chunks_written,
    )
    return result


def _finalise(run: IngestionRun, result: IngestionResult) -> None:
    run.status = result.status
    run.finished_at = timezone.now()
    run.papers_seen = result.seen
    run.papers_created = result.created
    run.papers_updated = result.updated
    run.papers_skipped = result.skipped
    run.chunks_indexed = result.chunks_written
    run.papers_split = result.papers_split
    # Only advance the watermark on a run that reached the end. Advancing it
    # after a failure would skip everything the failed run never got to.
    if result.status == IngestionRun.Status.SUCCESS:
        run.watermark = result.watermark
    run.error_message = "\n".join(result.errors[:20])
    run.save()
