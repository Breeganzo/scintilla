"""URL routing for the papers app.

A DRF router generates list and detail routes from each viewset, which keeps
URL structure consistent as endpoints are added.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from papers.views import EvaluationRunViewSet, IngestionRunViewSet, PaperViewSet

router = DefaultRouter()
router.register("papers", PaperViewSet, basename="paper")
router.register("ingestion-runs", IngestionRunViewSet, basename="ingestion-run")
router.register("evaluation-runs", EvaluationRunViewSet, basename="evaluation-run")

urlpatterns = [
    path("", include(router.urls)),
]
