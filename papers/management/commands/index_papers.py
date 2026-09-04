"""Embed chunks and write them to OpenSearch.

Two modes:

``index_papers``
    Incremental. Embeds only chunks with no ``indexed_at`` - which ingestion
    clears on exactly the papers whose text changed - and writes them to the
    index the alias currently points at.

``index_papers --rebuild``
    Builds a brand new timestamped index from PostgreSQL, then moves the alias
    onto it in one atomic call. Needed whenever the mapping or the embedding
    model changes, because neither can be altered in place.

The old index is left on disk after a rebuild rather than deleted. If the new
one turns out to be wrong, moving the alias back is one command; recovering a
deleted index is a full re-embed.
"""

from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ingestion.indexing import (
    DEFAULT_BULK_SIZE,
    IndexingError,
    SearchIndex,
    build_client,
    index_corpus,
)


class Command(BaseCommand):
    help = "Embed chunks and index them into OpenSearch."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--all",
            action="store_true",
            help="Re-index every chunk, not only those never indexed.",
        )
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Build a new index from scratch and switch the alias onto it.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Stop after this many chunks. Useful for a first smoke test.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BULK_SIZE,
            help=f"Documents per bulk request (default {DEFAULT_BULK_SIZE}).",
        )

    def handle(self, *args, **options) -> None:
        rebuild: bool = options["rebuild"]
        search_index = SearchIndex(build_client())

        try:
            if rebuild:
                target = search_index.versioned_name()
                search_index.create_index(target)
                self.stdout.write(f"Building {target}")
            else:
                target = search_index.resolve_write_index()
                self.stdout.write(f"Writing to {target} (alias {search_index.alias})")

            result = index_corpus(
                search_index=search_index,
                index=target,
                # A rebuild starts from an empty index, so "already indexed"
                # says nothing about whether the new index holds the document.
                only_stale=not (rebuild or options["all"]),
                limit=options["limit"],
                batch_size=options["batch_size"],
            )
        except IndexingError as exc:
            raise CommandError(str(exc)) from exc

        if rebuild:
            previous = search_index.point_alias_at(target)
            self.stdout.write(f"Alias {search_index.alias} -> {target}")
            if previous:
                self.stdout.write(
                    f"Previous index kept: {', '.join(previous)} "
                    f"(delete once the new one is verified)"
                )

        self.stdout.write(
            f"seen={result.documents_seen} "
            f"indexed={result.documents_indexed} "
            f"failed={result.documents_failed} "
            f"papers_marked={result.papers_marked}"
        )
        self.stdout.write(f"index now holds {search_index.count(target)} documents")
        self.stdout.write(f"model={settings.EMBEDDING_MODEL} dim={settings.EMBEDDING_DIM}")

        for failure in result.failures[:10]:
            self.stderr.write(f"  {failure}")
        if result.documents_failed:
            raise CommandError(
                f"{result.documents_failed} documents failed to index. "
                f"They were left unmarked and will be retried on the next run."
            )
