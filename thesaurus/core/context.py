"""Context window management — pruning old tool results.

When the conversation approaches the model's context limit, old tool
result content is replaced with a short placeholder. This frees tokens
for the next API call without breaking the tool_use ↔ tool_result
pairing that the API requires.
"""

from typing import Any

PLACEHOLDER = "[Tool output cleared]"
_PLACEHOLDER_LEN = len(PLACEHOLDER)


def prune_tool_results(
    messages: list[dict[str, Any]], protect_last_n_turns: int = 2,
) -> int:
    """Replace old tool result content with a placeholder.

    Walks backwards through *messages*, skips the last
    *protect_last_n_turns* user messages (recent context the model
    needs), then clears ``tool_result`` content blocks that are longer
    than the placeholder itself.

    Returns the estimated number of tokens saved (``len // 4``).
    """
    user_turns_seen = 0
    tokens_saved = 0

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        user_turns_seen += 1
        if user_turns_seen <= protect_last_n_turns:
            continue

        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if block.get("type") != "tool_result":
                continue
            if block.get("is_error"):
                continue
            text = block.get("content", "")
            if not isinstance(text, str):
                continue
            if text == PLACEHOLDER or len(text) <= _PLACEHOLDER_LEN:
                continue

            tokens_saved += len(text) // 4
            block["content"] = PLACEHOLDER

    return tokens_saved
