"""Configuration, read once from the environment and validated at the boundary.

Every value here is validated when the process starts rather than when it is
first used. A malformed timeout that surfaces as a confusing failure on the
first tool call, ten minutes into a session, is far more expensive than a
refusal to start.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_API_URL = "http://127.0.0.1:8001/api"

# Larger than it looks like it needs to be, on purpose. The first dense or
# hybrid query after a restart loads the embedding model on demand and takes
# roughly fifteen seconds; every query after it takes tens of milliseconds. A
# ten-second timeout would turn a cold start into an error exactly once per
# deployment, which is the least debuggable moment to produce one.
DEFAULT_TIMEOUT_S = 30.0


class ConfigError(ValueError):
    """Raised when the environment cannot be turned into valid settings."""


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for the MCP server."""

    api_base_url: str = DEFAULT_API_URL
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from the environment, refusing anything malformed."""
        source: Mapping[str, str] = os.environ if env is None else env

        url = source.get("SCINTILLA_API_URL", DEFAULT_API_URL).strip()
        if not url:
            raise ConfigError("SCINTILLA_API_URL is set but empty")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"SCINTILLA_API_URL must start with http:// or https://, got {url!r}")
        # Every path is joined against this, so a trailing slash here plus a
        # leading slash there silently produces a double slash. Normalise once.
        url = url.rstrip("/")

        raw_timeout = source.get("SCINTILLA_MCP_TIMEOUT")
        if raw_timeout is None:
            timeout = DEFAULT_TIMEOUT_S
        else:
            try:
                timeout = float(raw_timeout)
            except ValueError as exc:
                raise ConfigError(
                    f"SCINTILLA_MCP_TIMEOUT must be a number, got {raw_timeout!r}"
                ) from exc
            if timeout <= 0:
                raise ConfigError(f"SCINTILLA_MCP_TIMEOUT must be positive, got {timeout}")

        return cls(api_base_url=url, timeout_s=timeout)
