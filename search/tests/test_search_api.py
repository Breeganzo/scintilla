"""Tests for POST /api/search/ and the mode registry."""

from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from search.retrievers import (
    MODES,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    UnknownModeError,
    get_retriever,
)
from search.serializers import MAX_QUERY_LENGTH, MAX_TOP_K
from search.tests.conftest import StubRetriever, ranked
from search.views import hydrate

URL = "/api/search/"


@pytest.fixture
def client() -> APIClient:
    return APIClient()


@pytest.fixture
def stub_search(monkeypatch):
    """Replace retriever construction so the API tests exercise only the view."""

    def _install(results, name: str = "hybrid"):
        stub = StubRetriever(name, results)
        monkeypatch.setattr("search.views.get_retriever", lambda mode, **kw: stub)
        return stub

    return _install


class TestRegistry:
    def test_every_mode_resolves(self):
        for mode in MODES:
            assert get_retriever(mode).name == mode

    def test_modes_are_the_three_the_ablation_compares(self):
        assert set(MODES) == {"bm25", "dense", "hybrid"}

    def test_returns_the_right_classes(self):
        assert isinstance(get_retriever("bm25"), BM25Retriever)
        assert isinstance(get_retriever("dense"), DenseRetriever)
        assert isinstance(get_retriever("hybrid"), HybridRetriever)

    def test_unknown_mode_lists_the_valid_ones(self):
        with pytest.raises(UnknownModeError, match="bm25"):
            get_retriever("magic")

    def test_the_default_is_hybrid(self):
        assert get_retriever().name == "hybrid"


class TestValidation:
    def test_a_missing_query_is_rejected(self, db, client):
        response = client.post(URL, {"mode": "bm25"}, format="json")

        assert response.status_code == 400
        assert "query" in response.data

    @pytest.mark.parametrize("query", ["", "   "])
    def test_a_blank_query_is_rejected(self, db, client, query):
        response = client.post(URL, {"query": query}, format="json")

        assert response.status_code == 400

    def test_an_unknown_mode_is_rejected_before_any_retrieval(self, db, client):
        response = client.post(URL, {"query": "higgs", "mode": "magic"}, format="json")

        assert response.status_code == 400
        assert "mode" in response.data

    def test_top_k_is_capped(self, db, client):
        """An uncapped top_k sizes a database query and a response body."""
        response = client.post(URL, {"query": "higgs", "top_k": MAX_TOP_K + 1}, format="json")

        assert response.status_code == 400

    def test_top_k_must_be_positive(self, db, client):
        response = client.post(URL, {"query": "higgs", "top_k": 0}, format="json")

        assert response.status_code == 400

    def test_an_overlong_query_is_rejected(self, db, client):
        response = client.post(URL, {"query": "x" * (MAX_QUERY_LENGTH + 1)}, format="json")

        assert response.status_code == 400

    def test_get_is_not_allowed(self, db, client):
        assert client.get(URL).status_code == 405


class TestResponse:
    def test_returns_hydrated_results_in_rank_order(self, db, client, make_paper, stub_search):
        make_paper("2401.00001", title="Higgs coupling to tau leptons")
        make_paper("2401.00002", title="Neutrino oscillation parameters")
        stub_search(ranked("2401.00002", "2401.00001"))

        response = client.post(URL, {"query": "higgs"}, format="json")

        assert response.status_code == 200
        assert [item["arxiv_id"] for item in response.data["results"]] == [
            "2401.00002",
            "2401.00001",
        ]
        assert response.data["results"][0]["title"] == "Neutrino oscillation parameters"

    def test_the_envelope_echoes_the_request(self, db, client, make_paper, stub_search):
        make_paper("2401.00001")
        stub_search(ranked("2401.00001"))

        response = client.post(URL, {"query": "higgs", "mode": "bm25", "top_k": 3}, format="json")

        assert response.data["query"] == "higgs"
        assert response.data["mode"] == "bm25"
        assert response.data["top_k"] == 3
        assert response.data["count"] == 1

    def test_mode_defaults_to_hybrid(self, db, client, stub_search):
        stub_search([])

        response = client.post(URL, {"query": "higgs"}, format="json")

        assert response.data["mode"] == "hybrid"

    def test_reports_how_long_retrieval_took(self, db, client, stub_search):
        stub_search([])

        response = client.post(URL, {"query": "higgs"}, format="json")

        assert response.data["took_ms"] >= 0

    def test_per_retriever_detail_survives_to_the_response(
        self, db, client, make_paper, stub_search
    ):
        """This is the 'why this result' data the Day 11 frontend renders."""
        make_paper("2401.00001")
        from search.fusion import reciprocal_rank_fusion

        stub_search(reciprocal_rank_fusion({"bm25": ranked("2401.00001"), "dense": []}))

        response = client.post(URL, {"query": "higgs"}, format="json")

        assert response.data["results"][0]["debug"]["runs"]["bm25"]["rank"] == 1
        assert response.data["results"][0]["debug"]["rrf_k"] == 60

    def test_no_matches_is_an_empty_list_not_an_error(self, db, client, stub_search):
        stub_search([])

        response = client.post(URL, {"query": "topic absent from the corpus"}, format="json")

        assert response.status_code == 200
        assert response.data["results"] == []
        assert response.data["count"] == 0

    def test_diagnostics_are_included_for_hybrid(self, db, client, monkeypatch):
        hybrid = HybridRetriever(StubRetriever("bm25"), StubRetriever("dense"))
        monkeypatch.setattr("search.views.get_retriever", lambda mode, **kw: hybrid)

        response = client.post(URL, {"query": "higgs"}, format="json")

        assert response.data["diagnostics"]["rrf_k"] == 60

    def test_baselines_report_no_fusion_diagnostics(self, db, client, stub_search):
        stub_search([], name="bm25")

        response = client.post(URL, {"query": "higgs", "mode": "bm25"}, format="json")

        assert response.data["diagnostics"] == {}


class TestHydration:
    def test_preserves_retrieval_order_not_database_order(self, db, make_paper):
        """The database returns papers newest-first; retrieval order must win."""
        make_paper("2401.00001")
        make_paper("2401.00002")
        make_paper("2401.00003")

        hydrated = hydrate(ranked("2401.00003", "2401.00001", "2401.00002"))

        assert [item["arxiv_id"] for item in hydrated] == [
            "2401.00003",
            "2401.00001",
            "2401.00002",
        ]

    def test_uses_one_query_regardless_of_result_count(
        self, db, make_paper, django_assert_num_queries
    ):
        """Hydrating per result would be an N+1 the moment top_k grows."""
        for n in range(1, 6):
            make_paper(f"2401.0000{n}")

        with django_assert_num_queries(1):
            hydrate(ranked("2401.00001", "2401.00002", "2401.00003", "2401.00004"))

    def test_drops_a_result_whose_paper_is_missing(self, db, make_paper):
        """The index being ahead of the database must not render a blank row."""
        make_paper("2401.00001")

        hydrated = hydrate(ranked("2401.00001", "2401.99999"))

        assert [item["arxiv_id"] for item in hydrated] == ["2401.00001"]

    def test_empty_input(self, db):
        assert hydrate([]) == []


class TestSchema:
    def test_the_endpoint_appears_in_the_openapi_schema(self, db, client):
        """The frontend generates its types from this; an absent path breaks the build."""
        response = client.get("/api/schema/?format=json")

        assert response.status_code == 200
        assert "/api/search/" in response.data["paths"]
