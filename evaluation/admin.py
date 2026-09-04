"""Admin for evaluation runs.

Registered here rather than in ``papers`` because this app owns the model that
the harness actually writes to.
"""

from django.contrib import admin

from evaluation.models import EvaluationRun, QueryResult


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "mode", "top_k", "short_sha", "corpus_papers")
    list_filter = ("mode", "golden_set_version")
    readonly_fields = ("created_at",)

    @admin.display(description="Commit")
    def short_sha(self, obj: EvaluationRun) -> str:
        return obj.git_sha[:7]


@admin.register(QueryResult)
class QueryResultAdmin(admin.ModelAdmin):
    list_display = ("query_id", "query_class", "run")
    list_filter = ("query_class",)
    raw_id_fields = ("run",)
