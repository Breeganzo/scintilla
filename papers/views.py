"""Read-only API endpoints.

Every viewset here is read-only by design. Papers enter the system through the
ingestion pipeline, which enforces the content-hash check and records an
IngestionRun. A writeable HTTP endpoint would be a second path into the corpus
that bypasses both - and a corpus that can be modified out of band is one
whose evaluation numbers cannot be trusted.
"""

from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import viewsets

from papers.models import EvaluationRun, IngestionRun, Paper
from papers.serializers import (
    EvaluationRunSerializer,
    IngestionRunSerializer,
    PaperDetailSerializer,
    PaperListSerializer,
)


@extend_schema_view(
    list=extend_schema(summary="List papers", tags=["papers"]),
    retrieve=extend_schema(summary="Retrieve one paper by arXiv ID", tags=["papers"]),
)
class PaperViewSet(viewsets.ReadOnlyModelViewSet):
    """Browse the indexed corpus."""

    queryset = Paper.objects.all()
    lookup_field = "arxiv_id"
    # arXiv IDs contain dots and slashes, which the default lookup regex
    # rejects. Without this, /api/papers/2401.12345v1/ returns 404.
    lookup_value_regex = "[^/]+"

    def get_serializer_class(self):
        if self.action == "retrieve":
            return PaperDetailSerializer
        return PaperListSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        if self.action == "retrieve":
            # Detail view reports chunk_count. Without prefetching, that is an
            # extra query per paper - harmless for one, an N+1 problem the
            # moment this is reused in a list.
            qs = qs.prefetch_related("chunks")
        category = self.request.query_params.get("category")
        if category:
            qs = qs.filter(primary_category=category)
        return qs


@extend_schema_view(
    list=extend_schema(summary="List ingestion runs", tags=["operations"]),
    retrieve=extend_schema(summary="Retrieve one ingestion run", tags=["operations"]),
)
class IngestionRunViewSet(viewsets.ReadOnlyModelViewSet):
    """Harvest history.

    Exposed over HTTP so that pipeline health is observable without database
    access, which matters once this is running somewhere other than a laptop.
    """

    queryset = IngestionRun.objects.all()
    serializer_class = IngestionRunSerializer


@extend_schema_view(
    list=extend_schema(summary="List evaluation runs", tags=["evaluation"]),
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
