"""Application configuration.

Uses pydantic-settings to read configuration from environment variables and
the ``.env`` file. This is the single source of truth for all settings —
no scattered ``os.environ.get()`` calls throughout the codebase.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Agent settings — populated from environment variables and .env file.

    Env vars take precedence over .env values. Pydantic validates types
    and raises a clear error if required fields are missing.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
    )

    anthropic_api_key: str
    model: str = "claude-sonnet-4-20250514"
    debug: bool = False
    max_turns: int = 10
    log_dir: Path = Path(__file__).resolve().parent.parent / "logs"


def get_settings() -> Settings:
    """Create and return the settings instance.

    Raises ``ValidationError`` if required fields (like ANTHROPIC_API_KEY)
    are missing.
    """
    return Settings()
