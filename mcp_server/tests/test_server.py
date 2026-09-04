"""Tool-level tests: schema shape, result mapping and failure behaviour."""

import json

import anyio
import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.config import Settings
from mcp_server.server import build_server

SETTINGS = Settings(api_base_url="https://api.test/api", timeout_s=5.0)


def _json_response(status: int, payload) -> httpx.Response:
    return httpx.Response(
        status,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def _invoke(handler, tool: str, arguments: dict):
    server = build_server(SETTINGS, transport=httpx.MockTransport(handler))

    async def go():
        return await server.call_tool(tool, arguments)

    return anyio.run(go)


def _payload(result):
    """Pull the structured dict back out of a tool result."""
    assert not result.is_error
    return result.structured_content


def _tools():
    server = build_server(
        SETTINGS, transport=httpx.MockTransport(lambda r: _json_response(200, {}))
    )

    async def go():
        return await server.list_tools()

    return anyio.run(go)


def _tool(name: str):
    return next(t for t in _tools() if t.name == name)


def test_exactly_three_tools_are_registered():
    assert sorted(t.name for t in _tools()) == [
        "get_paper",
        "retrieval_report",
        "search_papers",
    ]


def test_search_schema_constrains_mode_to_the_three_measured_modes():
    # The model picks the mode from this schema. An open string would invite
    # invented values and a round trip that can only fail.
    mode = _tool("search_papers").input_schema["properties"]["mode"]
    enum = mode.get("enum") or mode["allOf"][0]["enum"]
    assert sorted(enum) == ["bm25", "dense", "hybrid"]


def test_search_tool_description_warns_that_hybrid_is_not_better():
    # The finding contradicts the architecture, so it has to be where the model
    # actually reads: in the schema, before it chooses arguments.
    assert "NOT significantly better" in json.dumps(_tool("search_papers").input_schema)


def test_search_maps_results_and_explains_each_one():
    def handler(_request):
        return _json_response(
            200,
            {
                "query": "beam luminosity",
                "mode": "hybrid",
                "count": 1,
                "took_ms": 42.5,
                "diagnostics": {"rrf_k": 60, "degraded": []},
                "results": [
                    {
                        "rank": 1,
                        "arxiv_id": "2606.20830",
                        "title": "When CPT Violation Hides in Plain Sight",
                        "authors": ["A. Author"],
                        "primary_category": "hep-ph",
                        "published_at": "2026-06-01T00:00:00Z",
                        "abs_url": "https://arxiv.org/abs/2606.20830",
                        "debug": {
                            "rrf_k": 60,
                            "runs": {"bm25": {"rank": 1, "score": 9.63, "contribution": 0.0164}},
                        },
                    }
                ],
            },
        )

    payload = _payload(_invoke(handler, "search_papers", {"query": "beam luminosity"}))

    assert payload["count"] == 1
    paper = payload["results"][0]
    assert paper["arxiv_id"] == "2606.20830"
    assert paper["url"] == "https://arxiv.org/abs/2606.20830"
    assert "Only one retriever found this paper" in paper["why_this_ranked_here"]


def test_degraded_search_carries_a_warning():
    def handler(_request):
        return _json_response(
            200,
            {
                "query": "q",
                "mode": "hybrid",
                "count": 0,
                "took_ms": 1.0,
                "diagnostics": {"rrf_k": 60, "degraded": ["bm25"]},
                "results": [],
            },
        )

    payload = _payload(_invoke(handler, "search_papers", {"query": "q"}))

    assert "warning" in payload
    assert "bm25" in payload["warning"]


def test_healthy_search_carries_no_warning():
    def handler(_request):
        return _json_response(
            200,
            {
                "query": "q",
                "mode": "dense",
                "count": 0,
                "took_ms": 1.0,
                "diagnostics": {},
                "results": [],
            },
        )

    assert "warning" not in _payload(_invoke(handler, "search_papers", {"query": "q"}))


def test_throttling_surfaces_as_an_actionable_tool_error():
    def handler(_request):
        return _json_response(429, {"detail": "Request was throttled."})

    with pytest.raises(ToolError) as exc:
        _invoke(handler, "search_papers", {"query": "q"})
    assert "throttled" in str(exc.value)


def test_missing_paper_explains_that_the_corpus_is_a_subset():
    # A valid arXiv id that is simply not in this corpus is the common case,
    # and a bare 404 invites the model to conclude the paper does not exist.
    def handler(_request):
        return _json_response(404, {"detail": "Not found."})

    with pytest.raises(ToolError) as exc:
        _invoke(handler, "get_paper", {"arxiv_id": "9999.99999"})
    assert "fixed subset" in str(exc.value)


def test_retrieval_report_keeps_only_the_newest_run_per_mode():
    def handler(_request):
        return _json_response(
            200,
            {
                "results": [
                    {"mode": "dense", "git_sha": "new", "metrics": {"ndcg@10": 0.691}},
                    {"mode": "dense", "git_sha": "old", "metrics": {"ndcg@10": 0.5}},
                    {"mode": "bm25", "git_sha": "new", "metrics": {"ndcg@10": 0.473}},
                ]
            },
        )

    payload = _payload(_invoke(handler, "retrieval_report", {}))

    assert len(payload["runs"]) == 2
    dense = next(r for r in payload["runs"] if r["mode"] == "dense")
    assert dense["git_sha"] == "new"


def test_retrieval_report_states_that_hybrid_is_not_significantly_better():
    def handler(_request):
        return _json_response(200, {"results": [{"mode": "dense", "metrics": {}}]})

    payload = _payload(_invoke(handler, "retrieval_report", {}))
    assert "p = 0.194" in payload["how_to_read_this"]


def test_no_recorded_runs_is_reported_as_unmeasured_not_as_zero():
    # "No measurement exists" and "quality is zero" are different facts, and
    # returning an empty list without saying so conflates them.
    def handler(_request):
        return _json_response(200, {"results": []})

    payload = _payload(_invoke(handler, "retrieval_report", {}))

    assert payload["runs"] == []
    assert "unmeasured" in payload["note"]
