"""Message construction helpers.

Provides clean factory functions for building the message dicts that the
Anthropic API expects. Avoids hand-building dicts with magic string keys
scattered across the codebase.
"""

from typing import Any


def user_message(content: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Build a user message."""
    return {"role": "user", "content": content}


def assistant_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an assistant message."""
    return {"role": "assistant", "content": content}


def tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    """Build a tool_result content block."""
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        result["is_error"] = True
    return result
