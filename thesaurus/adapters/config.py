"""Application configuration.

Uses pydantic-settings to read configuration from environment variables and
the ``.env`` file. This is the single source of truth for all settings —
no scattered ``os.environ.get()`` calls throughout the codebase.
"""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Agent settings — populated from environment variables and .env file.

    Env vars take precedence over .env values. Pydantic validates types
    and raises a clear error if required fields are missing.

    Set ``API_PROVIDER=bedrock`` to use AWS Bedrock instead of the direct
    Anthropic API. Bedrock mode uses AWS credentials (region, profile,
    or explicit keys) and ignores ``ANTHROPIC_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
    )

    api_provider: Literal["anthropic", "bedrock"] = "anthropic"
    anthropic_api_key: str | None = None
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 10
    prune_context_threshold: float = 0.8
    log_dir: Path = Path(__file__).resolve().parent.parent / "logs"

    # Confluence (optional — search tool only registers when all three are set)
    confluence_url: str | None = None
    confluence_email: str | None = None
    confluence_api_key: str | None = None

    # AWS Bedrock settings (only used when provider=bedrock)
    aws_region: str | None = None
    aws_profile: str | None = None
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_session_token: str | None = None

    @model_validator(mode="after")
    def _validate_provider_settings(self) -> "Settings":
        if self.api_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when API_PROVIDER=anthropic"
            )
        return self


def get_settings() -> Settings:
    """Create and return the settings instance.

    Raises ``ValidationError`` if required fields (like ANTHROPIC_API_KEY)
    are missing.
    """
    return Settings()
