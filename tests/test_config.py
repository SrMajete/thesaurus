"""Tests for Settings configuration."""

import pytest
from pydantic import ValidationError

from thesaurus.adapters.config import Settings


class TestSettings:
    def test_anthropic_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("API_PROVIDER", "anthropic")
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)  # type: ignore[call-arg]
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_anthropic_with_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.api_provider == "anthropic"
        assert s.anthropic_api_key == "sk-test"

    def test_bedrock_does_not_require_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PROVIDER", "bedrock")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.api_provider == "bedrock"

    def test_default_model(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PROVIDER", "bedrock")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.model.startswith("claude-")

    def test_prune_context_threshold_default(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PROVIDER", "bedrock")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.prune_context_threshold == 0.8

    def test_custom_values_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PROVIDER", "bedrock")
        monkeypatch.setenv("MAX_TURNS", "20")
        monkeypatch.setenv("PRUNE_CONTEXT_THRESHOLD", "0.5")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.max_turns == 20
        assert s.prune_context_threshold == 0.5


class TestGetSettings:
    def test_returns_settings_instance(self, monkeypatch) -> None:
        from thesaurus.adapters.config import get_settings
        monkeypatch.setenv("API_PROVIDER", "bedrock")
        # get_settings reads from .env; rely on whatever's there or tolerate
        s = get_settings()
        assert isinstance(s, Settings)
