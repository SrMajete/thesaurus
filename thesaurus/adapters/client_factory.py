"""LLM client factory.

Builds the ``LLMClient`` adapter from ``Settings``. Composition roots
(TUI, future web layer) call ``make_llm_client`` to get a port-typed
client without knowing which SDK is behind it.
"""

import logging

from anthropic import Anthropic, AsyncAnthropic, AsyncAnthropicBedrock

from thesaurus.core.ports import LLMClient
from .anthropic_llm import AnthropicLLMClient
from .config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTEXT_TOKENS = 200_000


def make_llm_client(settings: Settings) -> LLMClient:
    """Create the LLM client based on the configured provider."""
    if settings.api_provider == "bedrock":
        raw: AsyncAnthropic | AsyncAnthropicBedrock = AsyncAnthropicBedrock(
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
            aws_access_key=settings.aws_access_key,
            aws_secret_key=settings.aws_secret_key,
            aws_session_token=settings.aws_session_token,
        )
    else:
        raw = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return AnthropicLLMClient(client=raw, model=settings.model)


def fetch_max_context_tokens(settings: Settings) -> int:
    """Fetch the model's max input token limit from the Anthropic API.

    Uses a sync client for a one-time call at startup. Returns a safe
    default for Bedrock (no models endpoint) or on any error.
    """
    if settings.api_provider == "bedrock":
        return _DEFAULT_MAX_CONTEXT_TOKENS

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        model_info = client.models.retrieve(settings.model)
        limit = model_info.max_input_tokens
        if limit and limit > 0:
            return limit
    except Exception:
        logger.warning(
            "Could not fetch model context limit for %s, "
            "falling back to %d",
            settings.model,
            _DEFAULT_MAX_CONTEXT_TOKENS,
        )

    return _DEFAULT_MAX_CONTEXT_TOKENS
