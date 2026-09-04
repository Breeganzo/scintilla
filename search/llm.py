"""Language model providers, behind one interface.

**Why an interface for something with one implementation.** Not to allow
swapping vendors - that is a benefit, not a reason. The reason is that
answering has to be *testable*, and a test that calls a hosted model over the
network is not a test: it is slow, costs money, needs a secret in CI, and
returns different text every run, so any assertion strong enough to be useful
would be flaky. Every behaviour that matters here - does it abstain, does it
cite, does it cite something that was actually retrieved - is a property of the
prompt and the parsing, and both can be exercised exactly with a stub.

**Graceful degradation is a product decision, not an error path.** With no API
key the system still retrieves; it just does not answer. Search is the part with
measured numbers behind it, and it should not go offline because an optional
downstream service is unconfigured. What must never happen is silent
degradation - an answer that quietly stopped being grounded looks exactly like
one that never was - so an unavailable provider is reported in the response,
not hidden.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Long enough for a 70B model to produce a few hundred tokens, short enough that
# a hung request cannot pin a worker. Retrieval has already completed by this
# point, so a timeout degrades to search-only rather than failing the request.
DEFAULT_TIMEOUT_SECONDS = 30

# Deterministic. A grounded answer that changes between identical requests
# cannot be evaluated, and creativity is not the goal - the model is being asked
# to summarise text it was handed, not to write.
DEFAULT_TEMPERATURE = 0.0

# How many times a rate-limited or server-errored request is retried.
#
# **Why this is not optional.** The free tier allows 8,000 tokens per minute and
# one grounded prompt is roughly 2,400, so a fifty-query evaluation sweep hits
# the limit within seconds. Without backoff the sweep does not fail loudly - it
# returns a report in which most queries "produced no answer", which reads
# exactly like a cautious model and is really a measure of the rate limiter.
# That is the failure this project keeps arguing about: a plausible number that
# is measuring the wrong thing.
DEFAULT_MAX_RETRIES = 5

# Ceiling on a single wait, so a mis-parsed header cannot stall a sweep for
# minutes.
MAX_BACKOFF_SECONDS = 65.0

# Groq puts the wait in a header, but also in the error prose, and the two do
# not always both appear. The prose is parsed as a fallback.
_RETRY_HINT = re.compile(r"try again in ([0-9.]+)s")


def _retry_after(response: requests.Response, attempt: int) -> float:
    """How long to wait before retrying, preferring what the server asked for.

    Guessing with pure exponential backoff when the API has stated the exact
    remaining window is how a client ends up either hammering the endpoint or
    sleeping far longer than necessary.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    match = _RETRY_HINT.search(response.text or "")
    if match:
        return min(float(match.group(1)) + 0.5, MAX_BACKOFF_SECONDS)
    return min(2.0**attempt, MAX_BACKOFF_SECONDS)


class LLMError(RuntimeError):
    """The provider was configured but could not be reached or refused."""


class LLMUnavailable(LLMError):
    """No provider is configured. Expected, and handled by degrading."""


@dataclass(frozen=True)
class Completion:
    """One model response, plus what it cost to get."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a system and user prompt into text."""

    name: str

    @property
    def available(self) -> bool:
        """False when the provider cannot be used, e.g. an unset key."""
        ...

    def complete(self, system: str, user: str) -> Completion:
        """Return the model's reply, or raise :class:`LLMError`."""
        ...


class NullProvider:
    """The provider used when none is configured.

    It reports itself unavailable and raises if called anyway. Returning an
    empty string instead would be the dangerous choice: an empty answer is
    indistinguishable from a model that had nothing to say, and the abstention
    rate would silently become 100% for the wrong reason.
    """

    name = "null"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str) -> Completion:
        raise LLMUnavailable(
            "No LLM provider is configured. Set GROQ_API_KEY to enable answering; "
            "search works without it."
        )


class GroqProvider:
    """Groq's OpenAI-compatible chat completions endpoint."""

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        session: requests.Session | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key if api_key is not None else getattr(settings, "GROQ_API_KEY", "")
        self.model = model or getattr(settings, "LLM_MODEL", "openai/gpt-oss-120b")
        self.timeout = timeout
        self.temperature = temperature
        self.session = session or requests.Session()
        self.max_retries = max_retries

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> Completion:
        if not self.available:
            raise LLMUnavailable("GROQ_API_KEY is not set.")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise LLMError(f"Groq request failed: {exc}") from exc

            if response.status_code == 200:
                return self._parse(response.json())

            # 429 is rate limiting and 5xx is the provider having a bad moment;
            # both are transient. A 400 or 404 is a request this client will
            # never get right by asking again, so it fails immediately rather
            # than burning a minute of backoff on a decommissioned model name.
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_retries:
                # The body is included because Groq puts the actual reason
                # there - rate limit, decommissioned model, malformed request -
                # and without it every failure looks the same in a log.
                raise LLMError(f"Groq returned {response.status_code}: {response.text[:300]}")

            wait = _retry_after(response, attempt)
            logger.info(
                "Groq returned %s; retrying in %.1fs (attempt %s/%s)",
                response.status_code,
                wait,
                attempt + 1,
                self.max_retries,
            )
            time.sleep(wait)

        raise LLMError("Groq retries exhausted.")  # pragma: no cover - loop always returns

    def _parse(self, payload: dict) -> Completion:
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {payload}") from exc

        usage = payload.get("usage", {})
        return Completion(
            text=text,
            model=payload.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def get_provider() -> LLMProvider:
    """The configured provider, or the null one.

    Resolved per call rather than cached at import so a key added to the
    environment takes effect without a restart, and so tests can override
    settings normally.
    """
    provider = GroqProvider()
    if provider.available:
        return provider
    logger.info("No LLM provider configured; answering is disabled.")
    return NullProvider()


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "GROQ_URL",
    "Completion",
    "GroqProvider",
    "LLMError",
    "LLMProvider",
    "LLMUnavailable",
    "NullProvider",
    "get_provider",
]
