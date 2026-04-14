"""One-line previews of tool-call parameters for the CLI display.

Both CLI implementations (classic and rich) render the same summary
per tool, so the mapping lives here as a single source of truth.
Keys are ``ToolName`` members — not raw strings — so the link to the
tool enum is type-checkable.
"""

from typing import Any, Callable

from .tools.base import ToolName


SUMMARIZERS: dict[ToolName, Callable[[dict[str, Any]], str]] = {
    ToolName.RUN_BASH: lambda p: p.get("command", ""),
    ToolName.RUN_PYTHON: lambda p: " ".join(p.get("code", "").split()),
    ToolName.READ_FILE: lambda p: p.get("file_path", ""),
    ToolName.WRITE_FILE: lambda p: f"{p.get('file_path', '')} ({len(p.get('content', ''))} chars)",
    ToolName.EDIT_FILE: lambda p: f"{p.get('file_path', '')} ({len(p.get('old_text', ''))}→{len(p.get('new_text', ''))} chars)",
    ToolName.GLOB_FILES: lambda p: p.get("pattern", ""),
    ToolName.GREP_FILES: lambda p: p.get("pattern", ""),
    # make_plan and send_response are streamable tools: on_tool_start fires
    # as soon as the first streamable field has content. The ``reason`` field
    # is usually populated by then, but the model doesn't always write it
    # first in the JSON, so the CLI may see empty params. These fallbacks
    # ensure the tool header still reads naturally in that case.
    ToolName.MAKE_PLAN: lambda *_: "planning…",
    ToolName.SEND_RESPONSE: lambda *_: "responding…",
}


def summarize_params(name: str, params: dict[str, Any]) -> str:
    """Return a one-line preview of a tool call's params for the CLI display."""
    summarizer = SUMMARIZERS.get(name)
    return summarizer(params) if summarizer else str(params)[:100]
