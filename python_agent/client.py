"""API client factory.

Builds the async Anthropic client (direct API or AWS Bedrock) from
``Settings``. Kept out of the CLI so both UI implementations can share
one source of truth for how the client is constructed.
"""

import logging

from anthropic import Anthropic, AsyncAnthropic, AsyncAnthropicBedrock

from .config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTEXT_TOKENS = 200_000


def make_client(settings: Settings) -> AsyncAnthropic | AsyncAnthropicBedrock:
    """Create the API client based on the configured provider."""
    if settings.api_provider == "bedrock":
        return AsyncAnthropicBedrock(
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
            aws_access_key=settings.aws_access_key,
            aws_secret_key=settings.aws_secret_key,
            aws_session_token=settings.aws_session_token,
        )
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


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
