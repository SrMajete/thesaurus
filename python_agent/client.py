"""API client factory.

Builds the async Anthropic client (direct API or AWS Bedrock) from
``Settings``. Kept out of the CLI so both UI implementations can share
one source of truth for how the client is constructed.
"""

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

from .config import Settings


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
