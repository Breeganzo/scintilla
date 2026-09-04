"""API tests for the evaluation endpoint.

The gap these close: the endpoint previously read a different model from the
one the harness writes to, so it returned an empty list against a populated
database and nothing noticed. Every test here therefore writes through the
same model the harness uses and reads back through HTTP. A test that mocked
the queryset would have passed against the broken wiring too.
"""

import pytest
from django.urls import reverse

from evaluation.models import EvaluationRun

pytestmark = pytest.mark.django_db


def _make_run(mode: str = "dense", **overrides) -> EvaluationRun:
    defaults = {
        "golden_set_version": 1,
        "mode": mode,
        "top_k": 10,
        "corpus_papers": 3377,
        "corpus_chunks": 8000,
        "git_sha": "a" * 40,
        "metrics": {"overall": {"ndcg@10": 0.691, "recall@10": 0.744}},
    }
    return EvaluationRun.objects.create(**{**defaults, **overrides})


def test_endpoint_serves_runs_written_by_the_harness(client):
    # The regression test for the bug: the harness writes, the API must read
    # the same rows. Asserting a non-empty body is the whole point.
    _make_run()

    body = client.get(reverse("evaluation-run-list")).json()

    assert body["count"] == 1
    assert body["results"][0]["mode"] == "dense"
    assert body["results"][0]["metrics"]["overall"]["ndcg@10"] == 0.691


def test_run_carries_the_provenance_needed_to_reproduce_it(client):
    # A metric without its corpus size, golden set and commit is a rumour.
    _make_run()

    run = client.get(reverse("evaluation-run-list")).json()["results"][0]

    for field in ("git_sha", "golden_set_version", "corpus_papers", "corpus_chunks", "top_k"):
        assert field in run, f"{field} missing; the number cannot be reproduced without it"


def test_mode_filter_narrows_to_one_retriever(client):
    _make_run("bm25")
    _make_run("dense")

    body = client.get(reverse("evaluation-run-list"), {"mode": "bm25"}).json()

    assert body["count"] == 1
    assert body["results"][0]["mode"] == "bm25"


def test_newest_run_is_returned_first(client):
    older = _make_run(git_sha="b" * 40)
    newer = _make_run(git_sha="c" * 40)

    results = client.get(reverse("evaluation-run-list")).json()["results"]

    assert [r["id"] for r in results] == [newer.id, older.id]


def test_endpoint_is_read_only(client):
    # Evaluation numbers enter through the harness, which records the corpus
    # and commit they came from. A writeable endpoint would be a second path
    # that records neither.
    response = client.post(reverse("evaluation-run-list"), {"mode": "dense"})

    assert response.status_code == 405
