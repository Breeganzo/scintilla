"""Tests for the OpenSearch index: mapping, bulk writes, aliases, marking.

The mapping tests look pedantic. They are pinning the decisions that cannot be
changed after the index exists - a wrong vector dimension or a missing analyzer
is a full rebuild, and the failure it causes at query time never mentions the
mapping.
"""

from __future__ import annotations

import pytest
from django.core.management import call_command

from ingestion.indexing import (
    MAX_INDEXED_AUTHORS,
    IndexingError,
    SearchIndex,
    build_document,
    build_index_body,
    document_id,
    index_corpus,
)
from papers.models import EMBEDDING_DIMENSIONS, Chunk, Paper

from .conftest import FakeOpenSearch, make_source

pytestmark = pytest.mark.django_db


def make_paper(
    arxiv_id: str = "2401.00001",
    *,
    chunks: int = 1,
    authors: list[str] | None = None,
    indexed: bool = False,
) -> Paper:
    source = make_source(arxiv_id)
    paper = Paper.objects.create(
        arxiv_id=source.arxiv_id,
        version=source.version,
        title=source.title,
        abstract=source.abstract,
        authors=authors if authors is not None else source.authors,
        categories=source.categories,
        primary_category=source.primary_category,
        published_at=source.published,
        arxiv_updated_at=source.updated,
        abs_url=source.abs_url,
        pdf_url=source.pdf_url,
    )
    for position in range(chunks):
        Chunk.objects.create(
            paper=paper,
            chunk_index=position,
            text=f"{source.title} part {position}. {source.abstract}",
            token_count=60,
            embedding_model="hashing-test-embedder",
            embedding_dim=EMBEDDING_DIMENSIONS,
            embedding=[0.0] * (EMBEDDING_DIMENSIONS - 1) + [1.0] if indexed else None,
            indexed_at=paper.updated_at if indexed else None,
        )
    return paper


class TestMapping:
    def test_mapping_is_strict(self) -> None:
        """A field not in the mapping must raise, not be invented.

        Dynamic mapping would let a typo create a real field and index nothing
        into the intended one, producing a query that returns zero hits with no
        error to explain it.
        """
        assert build_index_body()["mappings"]["dynamic"] == "strict"

    def test_no_vector_field(self) -> None:
        """Vectors live in PostgreSQL, not here.

        Pinned so that a future change has to be deliberate: a knn_vector field
        needs the k-NN plugin, which has no macOS build, and reintroducing one
        would break the development environment rather than this test.
        """
        properties = build_index_body()["mappings"]["properties"]
        assert "embedding" not in properties
        assert "knn" not in build_index_body()["settings"]["index"]

    def test_single_shard_no_replicas(self) -> None:
        # A replica on a single node can never be allocated, which pins cluster
        # health at yellow and teaches you to ignore the health signal.
        settings_block = build_index_body()["settings"]["index"]
        assert settings_block["number_of_shards"] == 1
        assert settings_block["number_of_replicas"] == 0

    def test_text_fields_are_analysed_as_english(self) -> None:
        # Stemming is what makes "measuring decays" match "measurement of decay".
        properties = build_index_body()["mappings"]["properties"]
        for name in ("title", "abstract", "text"):
            assert properties[name] == {"type": "text", "analyzer": "english"}

    def test_facet_fields_are_keywords(self) -> None:
        properties = build_index_body()["mappings"]["properties"]
        for name in ("arxiv_id", "authors", "categories", "primary_category"):
            assert properties[name]["type"] == "keyword"


class TestDocumentBuilding:
    def test_document_id_is_deterministic(self) -> None:
        assert document_id("2401.00001", 0) == "2401.00001#0"

    def test_document_id_carries_the_chunk(self) -> None:
        # One chunk per paper today. Keying on arxiv_id alone would silently
        # overwrite chunk 0 with chunk 1 the first time a paper is split.
        assert document_id("2401.00001", 1) != document_id("2401.00001", 0)

    def test_author_list_is_capped(self) -> None:
        """hep-ex collaboration papers carry thousands of authors.

        The full list stays in PostgreSQL, which is what the API serves.
        Indexing all of them would inflate the index for a facet nobody uses
        past the first few names.
        """
        paper = make_paper(authors=[f"Author {n}" for n in range(3000)])
        document = build_document(paper.chunks.first())
        assert len(document["authors"]) == MAX_INDEXED_AUTHORS

    def test_document_holds_the_embedded_text_but_no_vector(self) -> None:
        paper = make_paper()
        chunk = paper.chunks.first()
        document = build_document(chunk)
        assert document["text"] == chunk.text
        assert "embedding" not in document


class TestBulkIndexing:
    def test_documents_land_in_the_index(self, search_index, fake_opensearch) -> None:
        indexed, failures = search_index.bulk_index(
            "papers-test-1", [("a#0", {"arxiv_id": "a"}), ("b#0", {"arxiv_id": "b"})]
        )
        assert (indexed, failures) == (2, [])
        assert set(fake_opensearch.documents["papers-test-1"]) == {"a#0", "b#0"}

    def test_one_request_per_batch(self, search_index, fake_opensearch) -> None:
        search_index.bulk_index(
            "papers-test-1", [(f"d{n}#0", {"arxiv_id": f"d{n}"}) for n in range(50)]
        )
        assert len(fake_opensearch.bulk_calls) == 1

    def test_per_item_errors_are_detected(self, search_index) -> None:
        """The bulk API returns HTTP 200 with failures buried in the body.

        Trusting the status code is how an index ends up quietly missing part
        of the corpus, with a lower recall number as the only symptom.
        """
        client = FakeOpenSearch(fail_ids={"b#0"}, retry_fail_ids={"b#0"})
        index = SearchIndex(client, alias="papers-test")
        indexed, failures = index.bulk_index(
            "papers-test-1",
            [("a#0", {"arxiv_id": "a"}), ("b#0", {"arxiv_id": "b"}), ("c#0", {"arxiv_id": "c"})],
        )
        assert indexed == 2
        assert len(failures) == 1
        assert "b#0" in failures[0]

    def test_failed_item_is_retried_alone(self, search_index) -> None:
        """A batch can fail for reasons that are not one document's fault.

        Retrying individually is what separates "one malformed record" from
        "the whole batch was rejected".
        """
        client = FakeOpenSearch(fail_ids={"b#0"})
        index = SearchIndex(client, alias="papers-test")
        indexed, failures = index.bulk_index(
            "papers-test-1", [("a#0", {"arxiv_id": "a"}), ("b#0", {"arxiv_id": "b"})]
        )
        assert (indexed, failures) == (2, [])
        assert "b#0" in client.documents["papers-test-1"]

    def test_empty_batch_makes_no_request(self, search_index, fake_opensearch) -> None:
        assert search_index.bulk_index("papers-test-1", []) == (0, [])
        assert fake_opensearch.bulk_calls == []


class TestAlias:
    def test_first_run_bootstraps_index_and_alias(self, search_index, fake_opensearch) -> None:
        # A fresh install has neither, so setup can be "run the indexer" rather
        # than a sequence of curl commands the reader has to get right.
        target = search_index.resolve_write_index()
        assert target.startswith("papers-test-")
        assert fake_opensearch.aliases[target] == {"papers-test"}

    def test_existing_alias_is_reused(self, search_index) -> None:
        first = search_index.resolve_write_index()
        assert search_index.resolve_write_index() == first

    def test_alias_moves_in_a_single_request(self, search_index, fake_opensearch) -> None:
        """Remove and add must travel together.

        As two calls there is an instant where the alias resolves to nothing,
        and every reader in that window gets index_not_found_exception.
        """
        old = search_index.resolve_write_index()
        fake_opensearch.alias_calls.clear()
        new = "papers-test-20990101000000"
        search_index.create_index(new)
        previous = search_index.point_alias_at(new)

        assert previous == [old]
        assert len(fake_opensearch.alias_calls) == 1
        actions = fake_opensearch.alias_calls[0]["actions"]
        assert {"remove": {"index": old, "alias": "papers-test"}} in actions
        assert {"add": {"index": new, "alias": "papers-test"}} in actions
        assert fake_opensearch.aliases[new] == {"papers-test"}

    def test_concrete_index_using_the_alias_name_is_refused(
        self, search_index, fake_opensearch
    ) -> None:
        # It would block the alias from ever being created, so a rebuild could
        # never swap atomically. Better to say so than to half-work.
        fake_opensearch.indices.create("papers-test")
        with pytest.raises(IndexingError, match="not an alias"):
            search_index.resolve_write_index()

    def test_creating_an_existing_index_is_a_no_op(self, search_index, fake_opensearch) -> None:
        search_index.create_index("papers-test-1")
        search_index.create_index("papers-test-1")
        assert fake_opensearch.created == ["papers-test-1"]


class TestOrphanedChunks:
    def test_extra_chunks_are_dropped_and_chunk_zero_survives(
        self, search_index, fake_opensearch
    ) -> None:
        """A three-chunk paper revised down to one leaves #1 and #2 behind.

        Nothing overwrites them, so they keep answering queries with text the
        paper no longer contains. Chunk 0 is exempt because it always exists
        and is always rewritten, which keeps the paper searchable throughout.
        """
        target = "papers-test-1"
        fake_opensearch.documents[target] = {
            "2401.00001#0": {"arxiv_id": "2401.00001", "chunk_index": 0},
            "2401.00001#1": {"arxiv_id": "2401.00001", "chunk_index": 1},
            "2401.00001#2": {"arxiv_id": "2401.00001", "chunk_index": 2},
            "2401.99999#1": {"arxiv_id": "2401.99999", "chunk_index": 1},
        }
        search_index.drop_extra_chunks(target, ["2401.00001"])
        assert set(fake_opensearch.documents[target]) == {"2401.00001#0", "2401.99999#1"}

    def test_no_papers_means_no_request(self, search_index, fake_opensearch) -> None:
        search_index.drop_extra_chunks("papers-test-1", [])
        assert fake_opensearch.deleted_by_query == []


class TestIndexCorpus:
    def test_chunks_are_embedded_and_indexed(self, search_index, fake_opensearch, embedder) -> None:
        make_paper("2401.00001")
        make_paper("2401.00002")
        result = index_corpus(embedder=embedder, search_index=search_index)

        assert result.documents_indexed == 2
        assert result.ok
        assert search_index.count(result.index) == 2

    def test_indexed_vectors_are_normalised(self, search_index, embedder) -> None:
        paper = make_paper()
        index_corpus(embedder=embedder, search_index=search_index)
        chunk = paper.chunks.first()
        chunk.refresh_from_db()
        length = sum(float(value) ** 2 for value in chunk.embedding) ** 0.5
        assert length == pytest.approx(1.0)

    def test_vector_is_stored_in_postgres(self, search_index, embedder) -> None:
        paper = make_paper()
        index_corpus(embedder=embedder, search_index=search_index)
        chunk = paper.chunks.first()
        chunk.refresh_from_db()
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS

    def test_success_is_recorded_in_postgres(self, search_index, embedder) -> None:
        paper = make_paper()
        index_corpus(embedder=embedder, search_index=search_index)

        paper.refresh_from_db()
        assert paper.indexed_at is not None
        assert paper.chunks.filter(indexed_at__isnull=True).count() == 0

    def test_second_pass_indexes_nothing(self, search_index, fake_opensearch, embedder) -> None:
        """The same gate as ingestion, one layer down.

        Re-indexing an unchanged corpus is the expensive half of the pipeline -
        every document would be re-embedded to produce byte-identical vectors.
        """
        make_paper("2401.00001")
        make_paper("2401.00002")
        index_corpus(embedder=embedder, search_index=search_index)
        fake_opensearch.bulk_calls.clear()

        second = index_corpus(embedder=embedder, search_index=search_index)
        assert second.documents_seen == 0
        assert fake_opensearch.bulk_calls == []

    def test_all_forces_a_re_index(self, search_index, embedder) -> None:
        make_paper()
        index_corpus(embedder=embedder, search_index=search_index)
        second = index_corpus(embedder=embedder, search_index=search_index, only_stale=False)
        assert second.documents_indexed == 1

    def test_failures_leave_chunks_unmarked(self, embedder) -> None:
        """Marking a document that failed loses it permanently.

        Re-indexing one that actually succeeded is free, so on any doubt the
        chunk stays stale and the next run retries it.
        """
        paper = make_paper("2401.00001")
        client = FakeOpenSearch(fail_ids={"2401.00001#0"}, retry_fail_ids={"2401.00001#0"})
        index = SearchIndex(client, alias="papers-test")

        result = index_corpus(embedder=embedder, search_index=index)

        assert result.documents_failed == 1
        assert not result.ok
        paper.refresh_from_db()
        assert paper.indexed_at is None
        assert paper.chunks.filter(indexed_at__isnull=True).count() == 1

    def test_paper_split_across_chunks_is_marked_only_when_complete(
        self, search_index, embedder
    ) -> None:
        """A paper is not searchable in full until its last chunk lands.

        Marking it after the first batch would claim coverage the index does
        not have.
        """
        paper = make_paper("2401.00001", chunks=3)
        index_corpus(embedder=embedder, search_index=search_index, batch_size=2)

        paper.refresh_from_db()
        assert paper.indexed_at is not None
        assert paper.chunks.count() == 3

    def test_limit_stops_early(self, search_index, embedder) -> None:
        make_paper("2401.00001")
        make_paper("2401.00002")
        result = index_corpus(embedder=embedder, search_index=search_index, limit=1)
        assert result.documents_indexed == 1

    def test_index_is_refreshed_so_reads_see_writes(
        self, search_index, fake_opensearch, embedder
    ) -> None:
        # Without this, verification races the ~1s auto-refresh and fails
        # intermittently, which is worse than failing consistently.
        make_paper()
        result = index_corpus(embedder=embedder, search_index=search_index)
        assert result.index in fake_opensearch.refreshes


class TestManagementCommand:
    def test_incremental_run(self, monkeypatch, search_index, embedder, capsys) -> None:
        make_paper()
        monkeypatch.setattr("ingestion.indexing.get_embedder", lambda: embedder)
        monkeypatch.setattr(
            "papers.management.commands.index_papers.SearchIndex", lambda client: search_index
        )
        monkeypatch.setattr("papers.management.commands.index_papers.build_client", lambda: None)

        call_command("index_papers")
        output = capsys.readouterr().out
        assert "indexed=1" in output

    def test_rebuild_switches_the_alias(
        self, monkeypatch, search_index, fake_opensearch, embedder, capsys
    ) -> None:
        make_paper()
        monkeypatch.setattr("ingestion.indexing.get_embedder", lambda: embedder)
        monkeypatch.setattr(
            "papers.management.commands.index_papers.SearchIndex", lambda client: search_index
        )
        monkeypatch.setattr("papers.management.commands.index_papers.build_client", lambda: None)

        call_command("index_papers")
        original = next(iter(fake_opensearch.aliases))

        call_command("index_papers", "--rebuild")
        live = [index for index, aliases in fake_opensearch.aliases.items() if aliases]

        assert live != [original]
        assert len(live) == 1
        # The old index is kept, not deleted. Rolling back is one command;
        # recovering a deleted index is a full re-embed.
        assert original in fake_opensearch.documents

    def test_rebuild_indexes_everything_not_only_stale(
        self, monkeypatch, search_index, embedder, capsys
    ) -> None:
        """A fresh index holds nothing, so "already indexed" says nothing about it."""
        make_paper(indexed=True)
        monkeypatch.setattr("ingestion.indexing.get_embedder", lambda: embedder)
        monkeypatch.setattr(
            "papers.management.commands.index_papers.SearchIndex", lambda client: search_index
        )
        monkeypatch.setattr("papers.management.commands.index_papers.build_client", lambda: None)

        call_command("index_papers", "--rebuild")
        assert "indexed=1" in capsys.readouterr().out
