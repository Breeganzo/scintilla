"""Django admin registration.

Worth the few minutes it costs: it gives a working interface for inspecting
ingestion and evaluation state without writing SQL, which is genuinely useful
while debugging a pipeline.
"""

from django.contrib import admin
from django.utils.html import format_html

from papers.models import Chunk, IngestionRun, Paper


@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = ("arxiv_id", "short_title", "primary_category", "published_at", "indexed")
    list_filter = ("primary_category", "published_at")
    search_fields = ("arxiv_id", "title", "abstract")
    readonly_fields = ("content_hash", "created_at", "updated_at")
    date_hierarchy = "published_at"

    @admin.display(description="Title")
    def short_title(self, obj: Paper) -> str:
        return obj.title[:70] + ("..." if len(obj.title) > 70 else "")

    @admin.display(description="Indexed", boolean=True)
    def indexed(self, obj: Paper) -> bool:
        return obj.indexed_at is not None


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("__str__", "embedding_model", "token_count", "indexed_at")
    list_filter = ("embedding_model",)
    search_fields = ("paper__arxiv_id", "text")
    raw_id_fields = ("paper",)


@admin.register(IngestionRun)
class IngestionRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "started_at",
        "status_badge",
        "papers_seen",
        "papers_created",
        "papers_updated",
        "papers_skipped",
        "triggered_by",
    )
    list_filter = ("status", "triggered_by")
    readonly_fields = ("started_at",)

    @admin.display(description="Status")
    def status_badge(self, obj: IngestionRun) -> str:
        colours = {
            obj.Status.SUCCESS: "#1a7f37",
            obj.Status.RUNNING: "#9a6700",
            obj.Status.PARTIAL: "#bc4c00",
            obj.Status.FAILED: "#cf222e",
        }
        return format_html(
            '<b style="color:{}">{}</b>',
            colours.get(obj.status, "#000"),
            obj.get_status_display(),
        )
