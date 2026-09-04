"""A thin HTTP client for the Scintilla API.

Failure is a value here, not an exception - the same rule the web client
follows. A tool implementation cannot reach the success path without first
proving it is on it, so "forgot to handle the error" is not expressible. That
matters more for an MCP server than for a browser: the caller is a language
model, and an unhandled exception reaching it as a stack trace invites it to
invent an answer rather than report that the corpus was unavailable.
"""

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from mcp_server.config import Settings

FailureKind = Literal[
    "unreachable",
    "timeout",
    "not_found",
    "throttled",
    "invalid",
    "server",
    "malformed",
]

SearchMode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class Ok:
    """A successful call and its decoded body."""

    value: Any


@dataclass(frozen=True)
class Err:
    """A failed call, classified so callers can say something useful."""

    kind: FailureKind
    message: str


ApiResult = Ok | Err


def _message_from_body(body: Any, fallback: str) -> str:
    """Unwrap a DRF error body into one readable line.

    DRF reports validation errors as ``{"field": ["problem", ...]}``. Rendering
    that dict verbatim to a language model is noise; naming the field and the
    problem is actionable.
    """
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        parts: list[str] = []
        for field, problems in body.items():
            if isinstance(problems, list):
                joined = "; ".join(str(p) for p in problems)
            else:
                joined = str(problems)
            parts.append(f"{field}: {joined}")
        if parts:
            return " | ".join(parts)
    if isinstance(body, str) and body:
        return body
    return fallback


def _classify(response: httpx.Response) -> Err | None:
    """Map an HTTP status onto a failure kind, or None if the call succeeded."""
    status = response.status_code
    if status < 400:
        return None

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    if status == 404:
        return Err("not_found", _message_from_body(body, "Not found."))
    if status == 429:
        return Err(
            "throttled",
            _message_from_body(
                body,
                "Rate limited. Search is capped at 60 requests per minute.",
            ),
        )
    if status < 500:
        return Err("invalid", _message_from_body(body, f"Request rejected ({status})."))
    return Err("server", _message_from_body(body, f"The API failed ({status})."))


class ScintillaClient:
    """Async client over the read-only Scintilla API."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.api_base_url,
            timeout=settings.timeout_s,
            transport=transport,
            headers={"User-Agent": "scintilla-mcp"},
        )

    async def __aenter__(self) -> "ScintillaClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException:
            return Err(
                "timeout",
                f"The API did not respond within {self._settings.timeout_s:g}s.",
            )
        except httpx.HTTPError as exc:
            return Err(
                "unreachable",
                f"Could not reach the API at {self._settings.api_base_url}: {exc}",
            )

        failure = _classify(response)
        if failure is not None:
            return failure

        try:
            return Ok(response.json())
        except ValueError:
            # A 200 that is not JSON usually means something in front of the
            # API answered instead - a proxy error page, or a tunnel that is
            # up but pointing at nothing.
            return Err("malformed", "The API returned a response that was not JSON.")

    async def search(self, query: str, mode: SearchMode, top_k: int) -> ApiResult:
        """Run one search. Bounds are enforced by the API, not duplicated here."""
        return await self._request(
            "POST",
            "/search/",
            json={"query": query, "mode": mode, "top_k": top_k},
        )

    async def get_paper(self, arxiv_id: str) -> ApiResult:
        """Fetch one paper by arXiv identifier."""
        return await self._request("GET", f"/papers/{arxiv_id}/")

    async def evaluation_runs(self, mode: SearchMode | None = None) -> ApiResult:
        """List recorded evaluation runs, newest first.

        List endpoints are paginated, so the envelope is unwrapped here. A
        caller that indexed into the raw body would silently read pagination
        metadata as data.
        """
        params = {"mode": mode} if mode else None
        result = await self._request("GET", "/evaluation-runs/", params=params)
        if isinstance(result, Err):
            return result

        body = result.value
        if isinstance(body, dict) and isinstance(body.get("results"), list):
            return Ok(body["results"])
        if isinstance(body, list):
            return Ok(body)
        return Err("malformed", "The evaluation-runs endpoint returned an unexpected shape.")
