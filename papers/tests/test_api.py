"""API tests."""

import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


class TestHealth:
    def test_reports_ok_when_database_is_reachable(self, client):
        response = client.get("/api/health/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["database"] == "ok"


class TestPaperEndpoints:
    def test_list_is_empty_initially(self, client):
        response = client.get("/api/papers/")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_list_returns_created_papers(self, client, paper):
        response = client.get("/api/papers/")
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["arxiv_id"] == paper.arxiv_id

    def test_list_omits_the_full_abstract(self, client, paper):
        """List responses carry a snippet, not the whole abstract."""
        result = client.get("/api/papers/").json()["results"][0]
        assert "abstract" not in result
        assert "abstract_snippet" in result

    def test_detail_lookup_handles_dotted_arxiv_ids(self, client, paper):
        """arXiv IDs contain dots, which the default DRF lookup regex rejects."""
        response = client.get(f"/api/papers/{paper.arxiv_id}/")
        assert response.status_code == 200
        assert response.json()["arxiv_id"] == "2401.12345v1"

    def test_detail_includes_indexing_state(self, client, paper):
        body = client.get(f"/api/papers/{paper.arxiv_id}/").json()
        assert body["needs_indexing"] is True
        assert body["indexed_at"] is None
        assert body["chunk_count"] == 0

    def test_unknown_paper_returns_404(self, client):
        assert client.get("/api/papers/9999.99999v1/").status_code == 404

    def test_filter_by_category(self, client, paper):
        assert client.get("/api/papers/?category=hep-ex").json()["count"] == 1
        assert client.get("/api/papers/?category=astro-ph").json()["count"] == 0

    def test_writes_are_rejected(self, client, paper_kwargs):
        """The corpus is populated by the pipeline, never over HTTP."""
        response = client.post("/api/papers/", data=paper_kwargs, content_type="application/json")
        assert response.status_code == 405


class TestSchema:
    def test_openapi_schema_is_served(self, client):
        """The frontend generates its TypeScript types from this endpoint."""
        response = client.get(reverse("schema"))
        assert response.status_code == 200

    def test_schema_documents_the_paper_endpoints(self, client):
        body = client.get(reverse("schema")).content.decode()
        assert "/api/papers/" in body
        assert "/api/health/" in body
