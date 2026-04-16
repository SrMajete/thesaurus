"""Tests for the API client factory and model limit fetcher."""

from unittest.mock import MagicMock, patch

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

from python_agent.client import (
    _DEFAULT_MAX_CONTEXT_TOKENS,
    fetch_max_context_tokens,
    make_client,
)
from python_agent.config import Settings


def _settings(**overrides) -> Settings:  # type: ignore[no-untyped-def]
    defaults: dict = {
        "api_provider": "anthropic",
        "anthropic_api_key": "sk-test",
        "model": "claude-test",
    }
    defaults.update(overrides)
    return Settings(**defaults, _env_file=None)  # type: ignore[call-arg]


class TestMakeClient:
    def test_anthropic_returns_async_client(self) -> None:
        c = make_client(_settings())
        assert isinstance(c, AsyncAnthropic)

    def test_bedrock_returns_bedrock_client(self) -> None:
        c = make_client(_settings(
            api_provider="bedrock",
            anthropic_api_key=None,
            aws_region="us-east-1",
        ))
        assert isinstance(c, AsyncAnthropicBedrock)


class TestFetchMaxContextTokens:
    def test_bedrock_returns_default(self) -> None:
        result = fetch_max_context_tokens(_settings(
            api_provider="bedrock", anthropic_api_key=None,
        ))
        assert result == _DEFAULT_MAX_CONTEXT_TOKENS

    def test_anthropic_success(self) -> None:
        mock_info = MagicMock()
        mock_info.max_input_tokens = 1_000_000
        mock_client = MagicMock()
        mock_client.models.retrieve.return_value = mock_info

        with patch("python_agent.client.Anthropic", return_value=mock_client):
            result = fetch_max_context_tokens(_settings())
        assert result == 1_000_000

    def test_anthropic_api_error_falls_back(self) -> None:
        mock_client = MagicMock()
        mock_client.models.retrieve.side_effect = Exception("network down")

        with patch("python_agent.client.Anthropic", return_value=mock_client):
            result = fetch_max_context_tokens(_settings())
        assert result == _DEFAULT_MAX_CONTEXT_TOKENS

    def test_anthropic_zero_limit_falls_back(self) -> None:
        mock_info = MagicMock()
        mock_info.max_input_tokens = 0
        mock_client = MagicMock()
        mock_client.models.retrieve.return_value = mock_info

        with patch("python_agent.client.Anthropic", return_value=mock_client):
            result = fetch_max_context_tokens(_settings())
        assert result == _DEFAULT_MAX_CONTEXT_TOKENS

    def test_anthropic_none_limit_falls_back(self) -> None:
        mock_info = MagicMock()
        mock_info.max_input_tokens = None
        mock_client = MagicMock()
        mock_client.models.retrieve.return_value = mock_info

        with patch("python_agent.client.Anthropic", return_value=mock_client):
            result = fetch_max_context_tokens(_settings())
        assert result == _DEFAULT_MAX_CONTEXT_TOKENS
