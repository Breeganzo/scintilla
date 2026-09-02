"""Domain models.

Four tables:

    Paper           arXiv metadata, plus the content hash that makes
                    re-ingestion idempotent
    Chunk           the unit that gets embedded and indexed
    IngestionRun    a ledger row per harvest, so run history is queryable
    EvaluationRun   retrieval metrics tied to the commit that produced them
"""

import hashlib

from django.db import models


class Paper(models.Model):
    """A single arXiv preprint.

    Only metadata is stored - title, abstract and bibliographic fields. Full
    PDF text is deliberately out of scope: parsing scientific PDFs reliably is
    a project in itself, and abstracts are enough to demonstrate and measure
    the retrieval behaviour this project is about.
    """

    arxiv_id = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="arXiv identifier including version, e.g. 2401.12345v2",
    )

    title = models.TextField()
    abstract = models.TextField()

    # Lists of strings. Postgres stores these as jsonb and can index into them;
    # a separate Author table would be more normalised but buys nothing here,
    # because authors are never queried independently of their paper.
    authors = models.JSONField(default=list)
    categories = models.JSONField(default=list)

    primary_category = models.CharField(max_length=32, db_index=True)

    published_at = models.DateTimeField(help_text="When the first version was submitted to arXiv")
    arxiv_updated_at = models.DateTimeField(help_text="When the most recent version was submitted")

    abs_url = models.URLField(max_length=500)
    pdf_url = models.URLField(max_length=500)

    content_hash = models.CharField(
        max_length=64,
        db_index=True,
        editable=False,
        help_text=(
            "SHA-256 of title + abstract. Ingestion compares this before doing "
            "any work: an unchanged hash means the paper can be skipped without "
            "re-embedding it. This is what makes the pipeline safe to re-run."
        ),
    )

    indexed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=(
            "When this paper was last written to the search index. NULL means "
            "it exists in Postgres but is not yet searchable - the two stores "
            "can legitimately disagree mid-run, and this field records that."
        ),
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-published_at"]
        indexes = [
            # Supports "recent papers in this category", the most common
            # browse query.
            models.Index(fields=["primary_category", "-published_at"]),
            # Supports "which papers still need indexing", which the ingestion
            # DAG runs on every pass.
            models.Index(fields=["indexed_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.arxiv_id}: {self.title[:60]}"

    def save(self, *args, **kwargs) -> None:
        # Recomputed on every save so the hash cannot drift from the content
        # it describes.
        self.content_hash = self.compute_content_hash(self.title, self.abstract)
        super().save(*args, **kwargs)

    @staticmethod
    def compute_content_hash(title: str, abstract: str) -> str:
        """Hash the fields that affect retrieval.

        Whitespace is normalised first so that a reformatted abstract with
        identical wording does not read as a change and trigger a pointless
        re-embed.

        Only title and abstract are hashed because only those are embedded and
        indexed. A corrected author list does not change search behaviour, so
        it should not cost a re-embed.
        """
        normalised = f"{' '.join(title.split())}\n\n{' '.join(abstract.split())}"
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    @property
    def needs_indexing(self) -> bool:
        """True if the search index is missing or stale for this paper."""
        return self.indexed_at is None or self.indexed_at < self.updated_at


class Chunk(models.Model):
    """A unit of text that gets embedded and indexed.

    Today there is exactly one chunk per paper, holding title + abstract
    together. An abstract is already a self-contained summary of roughly 150
    to 250 words, which sits comfortably inside the embedding model's context
    window - splitting it would separate a claim from the context that makes
    it meaningful, and would hurt retrieval rather than help it.

    The table nonetheless models a one-to-many relationship, so that adding
    full-text chunking later is new rows rather than a schema migration.
    """

    paper = models.ForeignKey(
        Paper,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    chunk_index = models.PositiveIntegerField(
        default=0,
        help_text="Position within the paper. Always 0 while one chunk per paper.",
    )

    text = models.TextField(help_text="Exactly the text that was embedded")

    token_count = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Recorded to detect silent truncation by the embedding model",
    )

    embedding_model = models.CharField(
        max_length=128,
        help_text=(
            "Which model produced the vector. Stored per chunk because a model "
            "change invalidates every vector produced by the previous one, and "
            "mixing them in a single index produces meaningless neighbours."
        ),
    )
    embedding_dim = models.PositiveIntegerField()

    indexed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["paper", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["paper", "chunk_index"],
                name="unique_chunk_per_paper_position",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.paper.arxiv_id}#{self.chunk_index}"


class IngestionRun(models.Model):
    """One harvest attempt.

    This is the ledger. It is the concrete reason for choosing a scheduler
    over cron: cron leaves a log file that has to be read by a human, whereas
    this table answers "when did ingestion last succeed", "what changed", and
    "is it silently failing" as ordinary queries.
    """

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        # Some records succeeded and some did not. Recorded distinctly because
        # treating a partial run as a success hides data loss, and treating it
        # as a failure triggers pointless retries of work already done.
        PARTIAL = "partial", "Partial"

    started_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.RUNNING,
        db_index=True,
    )

    categories = models.JSONField(
        default=list,
        help_text="arXiv categories requested for this run",
    )

    papers_seen = models.PositiveIntegerField(default=0)
    papers_created = models.PositiveIntegerField(default=0)
    papers_updated = models.PositiveIntegerField(default=0)
    papers_skipped = models.PositiveIntegerField(
        default=0,
        help_text="Content hash unchanged, so no work was needed",
    )
    chunks_indexed = models.PositiveIntegerField(default=0)

    error_message = models.TextField(blank=True)

    triggered_by = models.CharField(
        max_length=64,
        default="manual",
        help_text="scheduler, manual, backfill, test",
    )

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"Ingestion {self.pk} ({self.status}) {self.started_at:%Y-%m-%d %H:%M}"

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class EvaluationRun(models.Model):
    """Retrieval metrics from one pass over the golden set.

    Persisted rather than only written to a file so that quality over time is
    a query. Without history there is no way to answer "when did recall drop",
    only "recall is lower than I remember".
    """

    class Mode(models.TextChoices):
        BM25 = "bm25", "BM25 keyword"
        DENSE = "dense", "Dense vector"
        HYBRID = "hybrid", "Hybrid (RRF)"

    started_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    git_sha = models.CharField(
        max_length=40,
        db_index=True,
        help_text=(
            "The commit these numbers came from. A metric that cannot be traced "
            "to a specific revision cannot be reproduced, and a number that "
            "cannot be reproduced should not be quoted."
        ),
    )

    golden_set_version = models.CharField(
        max_length=32,
        help_text="Golden set file version. Comparing runs across different versions is invalid.",
    )

    mode = models.CharField(max_length=16, choices=Mode.choices, db_index=True)

    num_queries = models.PositiveIntegerField()

    metrics = models.JSONField(
        default=dict,
        help_text=(
            "Aggregate and per-class metrics: recall@k, MRR, nDCG@10. JSON "
            "rather than columns because the metric set is still changing, and "
            "a migration per new metric would discourage adding them."
        ),
    )

    is_baseline = models.BooleanField(
        default=False,
        help_text="Marks the run the CI regression gate compares against",
    )

    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["mode", "-started_at"]),
        ]

    def __str__(self) -> str:
        return f"Eval {self.mode} @ {self.git_sha[:7]} ({self.started_at:%Y-%m-%d})"
