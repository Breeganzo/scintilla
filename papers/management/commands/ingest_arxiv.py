"""Harvest arXiv into PostgreSQL.

A management command rather than a standalone script so it inherits Django's
settings, database connection and logging, and so it can be invoked identically
by a human, by a scheduler and by a container entrypoint.

    python manage.py ingest_arxiv --categories hep-ex --limit 200
    python manage.py ingest_arxiv --full-scan --limit 50
"""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils.dateparse import parse_datetime

from ingestion.embedding import get_token_counter
from ingestion.pipeline import ingest


class Command(BaseCommand):
    help = "Harvest papers from the arXiv API into the local corpus."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--categories",
            default=",".join(settings.ARXIV_CATEGORIES),
            help="Comma-separated arXiv categories. Defaults to ARXIV_CATEGORIES.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=settings.ARXIV_MAX_RESULTS,
            help="Maximum papers to fetch this run.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=100,
            help="Results per API request. arXiv caps this at 2000; smaller is more reliable.",
        )
        parser.add_argument(
            "--since",
            default=None,
            help="ISO-8601 timestamp. Overrides the stored watermark.",
        )
        parser.add_argument(
            "--full-scan",
            action="store_true",
            help=(
                "Ignore the watermark and re-scan from the beginning. Safe at any "
                "time: unchanged content hashes mean unchanged papers cost nothing."
            ),
        )
        parser.add_argument(
            "--triggered-by",
            default="manual",
            help="Recorded on the ingestion run: manual, scheduler, backfill.",
        )

    def handle(self, *args, **options) -> None:
        categories = [c.strip() for c in options["categories"].split(",") if c.strip()]
        if not categories:
            raise CommandError("At least one category is required.")

        since = None
        if options["since"]:
            since = parse_datetime(options["since"])
            if since is None:
                raise CommandError(f"Could not parse --since value: {options['since']!r}")

        result = ingest(
            categories=categories,
            limit=options["limit"],
            page_size=options["page_size"],
            since=since,
            use_watermark=not options["full_scan"],
            triggered_by=options["triggered_by"],
            # Count tokens with the embedding model's own tokenizer. Chunking
            # decides what the model is asked to embed, so counting with a
            # different tokenizer means the 512-token ceiling is enforced
            # against the wrong number and long abstracts get truncated.
            count_tokens=get_token_counter(),
        )

        style = self.style.SUCCESS if result.status == "success" else self.style.WARNING
        self.stdout.write(
            style(
                f"run={result.run_id} status={result.status} "
                f"seen={result.seen} created={result.created} updated={result.updated} "
                f"skipped={result.skipped} chunks={result.chunks_written} "
                f"split={result.papers_split}"
            )
        )

        if result.errors:
            self.stdout.write(self.style.ERROR(f"{len(result.errors)} record(s) failed:"))
            for message in result.errors[:10]:
                self.stdout.write(self.style.ERROR(f"  {message}"))
