"""Configuration is validated at the boundary, so test the boundary."""

import pytest

from mcp_server.config import DEFAULT_API_URL, DEFAULT_TIMEOUT_S, ConfigError, Settings


def test_defaults_when_environment_is_empty():
    settings = Settings.from_env({})
    assert settings.api_base_url == DEFAULT_API_URL
    assert settings.timeout_s == DEFAULT_TIMEOUT_S


def test_trailing_slash_is_normalised_away():
    # Paths are joined against this. Without normalisation the client would
    # request //search/, which some proxies answer with a redirect and others
    # with a 404.
    settings = Settings.from_env({"SCINTILLA_API_URL": "https://api.example.com/api/"})
    assert settings.api_base_url == "https://api.example.com/api"


def test_url_without_scheme_is_refused():
    with pytest.raises(ConfigError, match="http"):
        Settings.from_env({"SCINTILLA_API_URL": "api.example.com"})


def test_empty_url_is_refused():
    # Distinct from unset. An unset variable means "use the default"; a
    # variable set to the empty string means a deployment script produced
    # nothing, and defaulting would hide that.
    with pytest.raises(ConfigError, match="empty"):
        Settings.from_env({"SCINTILLA_API_URL": "   "})


def test_non_numeric_timeout_is_refused():
    with pytest.raises(ConfigError, match="must be a number"):
        Settings.from_env({"SCINTILLA_MCP_TIMEOUT": "thirty"})


@pytest.mark.parametrize("value", ["0", "-5"])
def test_non_positive_timeout_is_refused(value):
    with pytest.raises(ConfigError, match="must be positive"):
        Settings.from_env({"SCINTILLA_MCP_TIMEOUT": value})


def test_timeout_default_exceeds_cold_start():
    # The first dense query after a restart loads the embedding model and takes
    # roughly fifteen seconds. A default below that would fail exactly once per
    # deployment, at the least debuggable moment.
    assert DEFAULT_TIMEOUT_S > 15.0
