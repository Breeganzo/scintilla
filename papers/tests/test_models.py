"""Model tests.

The content-hash tests matter most: the entire idempotency guarantee of the
ingestion pipeline rests on that hash changing exactly when it should and
never when it should not.
"""

from datetime import timedelta

import pytest
from django.db.utils import IntegrityError
from django.utils import timezone

from papers.models import Chunk, IngestionRun, Paper


class TestContentHash:
    def test_is_deterministic(self):
        a = Paper.compute_content_hash("Title", "Abstract")
        b = Paper.compute_content_hash("Title", "Abstract")
        assert a == b

    def test_is_sha256_hex(self):
        digest = Paper.compute_content_hash("Title", "Abstract")
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_whitespace_is_normalised(self):
        """Reformatting must not read as a content change.

        arXiv abstracts arrive with inconsistent line wrapping. If wrapping
        affected the hash, every harvest would re-embed the entire corpus for
        no benefit.
        """
        tidy = Paper.compute_content_hash("Higgs boson", "We present a measurement.")
        messy = Paper.compute_content_hash(
            "  Higgs   boson  ", "We  present\n\na    measurement.  "
        )
        assert tidy == messy

    def test_changes_when_title_changes(self):
        before = Paper.compute_content_hash("Original title", "Abstract")
        after = Paper.compute_content_hash("Corrected title", "Abstract")
        assert before != after

    def test_changes_when_abstract_changes(self):
        before = Paper.compute_content_hash("Title", "Original abstract")
        after = Paper.compute_content_hash("Title", "Revised abstract")
        assert before != after

    def test_set_on_save(self, paper):
        assert paper.content_hash == Paper.compute_content_hash(paper.title, paper.abstract)

    def test_unaffected_by_non_indexed_fields(self, paper):
        """Only embedded fields are hashed.

        A corrected author list changes nothing about how the paper is
        retrieved, so it must not trigger a re-embed.
        """
        original = paper.content_hash
        paper.authors = ["Completely", "Different", "People"]
        paper.categories = ["astro-ph"]
        paper.save()
        paper.refresh_from_db()
        assert paper.content_hash == original


class TestNeedsIndexing:
    def test_true_when_never_indexed(self, paper):
        assert paper.indexed_at is None
        assert paper.needs_indexing is True

    def test_false_when_indexed_after_last_change(self, paper):
        # .update() bypasses auto_now, which would otherwise move updated_at
        # forward and make this assertion impossible to express.
        Paper.objects.filter(pk=paper.pk).update(indexed_at=timezone.now() + timedelta(minutes=1))
        paper.refresh_from_db()
        assert paper.needs_indexing is False

    def test_true_when_index_is_stale(self, paper):
        Paper.objects.filter(pk=paper.pk).update(indexed_at=timezone.now() - timedelta(days=1))
        paper.refresh_from_db()
        assert paper.needs_indexing is True


class TestPaperConstraints:
    def test_arxiv_id_is_unique(self, db, paper, paper_kwargs):
        with pytest.raises(IntegrityError):
            Paper.objects.create(**paper_kwargs)


class TestChunk:
    def test_position_is_unique_per_paper(self, paper):
        Chunk.objects.create(
            paper=paper,
            chunk_index=0,
            text="text",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )
        with pytest.raises(IntegrityError):
            Chunk.objects.create(
                paper=paper,
                chunk_index=0,
                text="different text",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding_dim=384,
            )

    def test_deleted_with_its_paper(self, paper):
        Chunk.objects.create(
            paper=paper,
            chunk_index=0,
            text="text",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )
        assert Chunk.objects.count() == 1
        paper.delete()
        assert Chunk.objects.count() == 0


class TestIngestionRun:
    def test_starts_in_running_state(self, db):
        run = IngestionRun.objects.create(categories=["hep-ex"])
        assert run.status == IngestionRun.Status.RUNNING
        assert run.finished_at is None

    def test_duration_is_none_until_finished(self, db):
        run = IngestionRun.objects.create(categories=["hep-ex"])
        assert run.duration_seconds is None

    def test_duration_computed_once_finished(self, db):
        run = IngestionRun.objects.create(categories=["hep-ex"])
        run.finished_at = run.started_at + timedelta(seconds=42)
        run.save()
        assert run.duration_seconds == pytest.approx(42.0)
