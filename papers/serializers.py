"""Serializers.

Two serializers for Paper: a light one for lists and a full one for detail.
Search results routinely return 20+ papers, and sending every abstract in full
for a result list is bandwidth spent on text the user has not asked to read.
"""

from rest_framework import serializers

from papers.models import Chunk, IngestionRun, Paper


class PaperListSerializer(serializers.ModelSerializer):
    """Light representation for result lists."""

    abstract_snippet = serializers.SerializerMethodField()

    class Meta:
        model = Paper
        fields = [
            "arxiv_id",
            "title",
            "authors",
            "primary_category",
            "categories",
            "published_at",
            "abs_url",
            "abstract_snippet",
        ]

    def get_abstract_snippet(self, obj: Paper) -> str:
        """First 300 characters, cut on a word boundary."""
        if len(obj.abstract) <= 300:
            return obj.abstract
        return obj.abstract[:300].rsplit(" ", 1)[0] + "..."


class PaperDetailSerializer(serializers.ModelSerializer):
    """Full representation, including indexing state."""

    chunk_count = serializers.IntegerField(source="chunks.count", read_only=True)
    needs_indexing = serializers.BooleanField(read_only=True)

    class Meta:
        model = Paper
        fields = [
            "arxiv_id",
            "version",
            "title",
            "abstract",
            "authors",
            "categories",
            "primary_category",
            "published_at",
            "arxiv_updated_at",
            "abs_url",
            "pdf_url",
            "content_hash",
            "indexed_at",
            "needs_indexing",
            "chunk_count",
            "created_at",
            "updated_at",
        ]


class ChunkSerializer(serializers.ModelSerializer):
    arxiv_id = serializers.CharField(source="paper.arxiv_id", read_only=True)

    class Meta:
        model = Chunk
        fields = [
            "id",
            "arxiv_id",
            "chunk_index",
            "text",
            "token_count",
            "embedding_model",
            "embedding_dim",
            "indexed_at",
        ]


class IngestionRunSerializer(serializers.ModelSerializer):
    duration_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = IngestionRun
        fields = [
            "id",
            "started_at",
            "finished_at",
            "duration_seconds",
            "status",
            "categories",
            "papers_seen",
            "papers_created",
            "papers_updated",
            "papers_skipped",
            "papers_split",
            "chunks_indexed",
            "watermark",
            "triggered_by",
            "error_message",
        ]
