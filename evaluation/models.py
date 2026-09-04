"""Persistence for evaluation runs.

A number without provenance is a rumour. Storing runs means "recall@10 was
0.71" can always be resolved into: which golden set, which retrieval mode,
which corpus, which commit, and which queries contributed. Day 9 writes into
these tables; Day 10 compares two rows to decide whether a change was a
regression.
"""

from django.db import models


class EvaluationRun(models.Model):
    """One sweep of one retrieval mode over one golden set."""

    class Mode(models.TextChoices):
        BM25 = "bm25", "Lexical (BM25)"
        DENSE = "dense", "Dense (pgvector)"
        HYBRID = "hybrid", "Hybrid (RRF)"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    golden_set_version = models.PositiveIntegerField()
    mode = models.CharField(max_length=16, choices=Mode.choices)
    top_k = models.PositiveIntegerField()

    # The corpus is not a constant. Relevance labels were assigned against a
    # snapshot, so a run against a different corpus size is not comparable to
    # an earlier one and the numbers must not be put in the same table.
    corpus_papers = models.PositiveIntegerField()
    corpus_chunks = models.PositiveIntegerField()

    # Which code produced this. Without it, a regression cannot be bisected.
    git_sha = models.CharField(max_length=40, blank=True)

    metrics = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["mode", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.mode} @ k={self.top_k} (golden v{self.golden_set_version})"


class QueryResult(models.Model):
    """What one query returned in one run, kept so a metric can be re-derived.

    Storing only the aggregate metric would make it impossible to answer "which
    query got worse?" after the fact, which is the only question that matters
    when a regression gate fires.
    """

    run = models.ForeignKey(EvaluationRun, related_name="results", on_delete=models.CASCADE)
    query_id = models.CharField(max_length=32)
    query_class = models.CharField(max_length=32, db_index=True)
    retrieved_ids = models.JSONField(default=list)
    relevant_ids = models.JSONField(default=list)
    metrics = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["query_id"]
        constraints = [
            models.UniqueConstraint(fields=["run", "query_id"], name="unique_query_per_run"),
        ]

    def __str__(self) -> str:
        return f"{self.query_id} ({self.run_id})"
