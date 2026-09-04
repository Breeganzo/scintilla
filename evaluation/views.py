"""Read-only API over recorded evaluation runs.

Published because a retrieval system that reports no quality numbers is asking
to be taken on trust. The numbers here are frequently lower than people expect;
that is the point of publishing them.
"""

from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import viewsets

from evaluation.models import EvaluationRun
from evaluation.serializers import EvaluationRunSerializer


@extend_schema_view(
    list=extend_schema(
        summary="List evaluation runs",
        tags=["evaluation"],
        parameters=[
            OpenApiParameter(
                name="mode",
                description="Filter to one retriever mode: bm25, dense or hybrid.",
                required=False,
                type=str,
                enum=[choice.value for choice in EvaluationRun.Mode],
            )
        ],
    ),
    retrieve=extend_schema(summary="Retrieve one evaluation run", tags=["evaluation"]),
)
class EvaluationRunViewSet(viewsets.ReadOnlyModelViewSet):
    """Retrieval quality over time, per retriever mode."""

    queryset = EvaluationRun.objects.all()
    serializer_class = EvaluationRunSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        mode = self.request.query_params.get("mode")
        if mode:
            qs = qs.filter(mode=mode)
        return qs
