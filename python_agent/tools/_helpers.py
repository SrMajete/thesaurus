"""Shared helpers for tool implementations.

Small utilities used by multiple tools — truncation, timeout clamping,
file validation, binary detection. Kept together because each is a few
lines and a file per helper would be worse than a utility grab-bag.
"""

from pathlib import Path

# ── Output truncation ─────────────────────────────────────────────────

MAX_OUTPUT_CHARS = 50_000


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate *text* to *max_chars*, keeping head and tail.

    When truncated, a marker in the middle shows the original length
    so the reader knows content was lost from the middle, not the end.
    """
    if len(text) <= max_chars:
        return text
    marker = f"\n\n... truncated ({len(text)} chars total) ...\n\n"
    half = (max_chars - len(marker)) // 2
    return text[:half] + marker + text[-half:]


# ── Timeout clamping ──────────────────────────────────────────────────

TIMEOUT_LIMIT = 600


def clamp_timeout(timeout: int) -> int:
    """Clamp a user-provided timeout to [1, TIMEOUT_LIMIT] seconds."""
    return max(1, min(timeout, TIMEOUT_LIMIT))


# ── File validation ───────────────────────────────────────────────────


def validate_file_path(path: Path) -> str | None:
    """Return an error string if *path* is not a readable file, else None."""
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"
    return None


def is_binary(content: bytes, check_bytes: int = 8192) -> bool:
    """Return True if *content* looks like a binary file (null bytes in prefix)."""
    return b"\x00" in content[:check_bytes]
