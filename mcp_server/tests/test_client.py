"""The client classifies failures. These tests check the classification.

Driven with ``anyio.run`` rather than a pytest async plugin: the project has no
async test dependency and adding one to test three methods would be a poor
trade. ``httpx.MockTransport`` supplies the responses, so real request building,
status handling and JSON decoding all run.
"""

import json
from functools import partial

import httpx

from mcp_server.client import Err, Ok, ScintillaClient
from mcp_server.config import Settings

try:
    import anyio
except ImportError:  # pragma: no cover - anyio ships with httpx and mcp
    anyio = None


SETTINGS = Settings(api_base_url="https://api.test/api", timeout_s=5.0)


def _run(coro_fn):
    return anyio.run(coro_fn)


def _client(handler) -> ScintillaClient:
    return ScintillaClient(SETTINGS, transport=httpx.MockTransport(handler))


def _call(handler, method_name, *args):
    async def go():
        async with _client(handler) as client:
            return await getattr(client, method_name)(*args)

    return _run(go)


def _json_response(status: int, payload) -> httpx.Response:
    return httpx.Response(
        status,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_search_posts_the_expected_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _json_response(200, {"results": [], "count": 0})

    result = _call(handler, "search", "beam luminosity", "dense", 10)

    assert isinstance(result, Ok)
    # No double slash: the base URL is normalised and the path adds exactly one.
    assert captured["url"] == "https://api.test/api/search/"
    assert captured["body"] == {"query": "beam luminosity", "mode": "dense", "top_k": 10}


def test_throttling_is_its_own_failure_kind():
    # A 429 is not a server error and must not read like one. It is temporary,
    # and the caller should be told to wait rather than to give up.
    handler = partial(lambda _req: _json_response(429, {"detail": "Request was throttled."}))
    result = _call(handler, "search", "q", "dense", 10)

    assert isinstance(result, Err)
    assert result.kind == "throttled"
    assert "throttled" in result.message.lower()


def test_validation_errors_name_the_field():
    def handler(_request):
        return _json_response(400, {"top_k": ["Ensure this value is less than or equal to 50."]})

    result = _call(handler, "search", "q", "dense", 500)

    assert isinstance(result, Err)
    assert result.kind == "invalid"
    assert "top_k" in result.message
    assert "50" in result.message


def test_missing_paper_is_not_found_rather_than_server_error():
    def handler(_request):
        return _json_response(404, {"detail": "No Paper matches the given query."})

    result = _call(handler, "get_paper", "9999.99999")

    assert isinstance(result, Err)
    assert result.kind == "not_found"


def test_server_error_is_classified_separately():
    def handler(_request):
        return _json_response(500, {"detail": "boom"})

    result = _call(handler, "search", "q", "dense", 10)

    assert isinstance(result, Err)
    assert result.kind == "server"


def test_connection_failure_is_reported_as_unreachable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    result = _call(handler, "search", "q", "dense", 10)

    assert isinstance(result, Err)
    assert result.kind == "unreachable"
    assert "https://api.test/api" in result.message


def test_timeout_is_distinct_from_unreachable():
    # Different remedies. Unreachable means the API is down; a timeout on this
    # system usually means a cold start is loading the embedding model.
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    result = _call(handler, "search", "q", "dense", 10)

    assert isinstance(result, Err)
    assert result.kind == "timeout"


def test_non_json_success_is_malformed_not_success():
    # A 200 that is not JSON means something in front of the API answered - a
    # proxy error page, or a tunnel that is up but pointing at nothing.
    def handler(_request):
        return httpx.Response(200, content="<html>gateway</html>")

    result = _call(handler, "search", "q", "dense", 10)

    assert isinstance(result, Err)
    assert result.kind == "malformed"


def test_evaluation_runs_unwraps_pagination():
    # List endpoints are paginated. A caller that indexed into the raw body
    # would read pagination metadata as data.
    def handler(_request):
        return _json_response(
            200,
            {"count": 1, "next": None, "previous": None, "results": [{"mode": "dense"}]},
        )

    result = _call(handler, "evaluation_runs", None)

    assert isinstance(result, Ok)
    assert result.value == [{"mode": "dense"}]


def test_evaluation_runs_rejects_an_unexpected_shape():
    def handler(_request):
        return _json_response(200, {"unexpected": True})

    result = _call(handler, "evaluation_runs", None)

    assert isinstance(result, Err)
    assert result.kind == "malformed"


def test_evaluation_runs_filters_by_mode():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return _json_response(200, {"results": []})

    _call(handler, "evaluation_runs", "hybrid")

    assert captured["params"] == {"mode": "hybrid"}
