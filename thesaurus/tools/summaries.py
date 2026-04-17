"""Per-tool display metadata shared by every CLI frontend.

- ``SUMMARIZERS`` — param → one-line preview string, used by tool
  headers and permission prompts.
- ``summarize_params`` — safe lookup wrapper for ``SUMMARIZERS``.
- ``tool_header_label`` — the ``{reason or summary}`` fallback that
  every CLI uses under its ``tool → {name}: …`` header. Centralised
  here so all three CLIs stay in sync if the fallback rule changes.
- ``INTERCEPTED_TOOLS`` — names of tools whose ``execute()`` is never
  called because the processor intercepts them. Derived from the
  ``is_intercepted`` flag on each tool so the registry stays the
  single source of truth.

Keyed by ``ToolName`` members (``StrEnum``, so they compare equal to
strings at runtime) — the link to the tool registry is type-checkable.
"""

from typing import Any, Callable

from . import get_default_tools
from .base import ToolName


SUMMARIZERS: dict[ToolName, Callable[[dict[str, Any]], str]] = {
    ToolName.RUN_BASH: lambda p: p.get("command", ""),
    ToolName.RUN_PYTHON: lambda p: " ".join(p.get("code", "").split()),
    ToolName.READ_FILE: lambda p: p.get("file_path", ""),
    ToolName.WRITE_FILE: lambda p: f"{p.get('file_path', '')} ({len(p.get('content', ''))} chars)",
    ToolName.EDIT_FILE: lambda p: f"{p.get('file_path', '')} ({len(p.get('old_text', ''))}→{len(p.get('new_text', ''))} chars)",
    ToolName.GLOB_FILES: lambda p: p.get("pattern", ""),
    ToolName.GREP_FILES: lambda p: p.get("pattern", ""),
    # make_plan and send_response are streamable tools: on_tool_start fires
    # as soon as the first streamable field has content, which may be before
    # the model has written the ``reason`` field. Return empty so the header
    # renders as just ``tool → make_plan`` / ``tool → send_response`` —
    # whatever streams next (thinking body, response body) is the real payload.
    ToolName.MAKE_PLAN: lambda *_: "",
    ToolName.SEND_RESPONSE: lambda *_: "",
    ToolName.FETCH_URL: lambda p: p.get("url", ""),
    ToolName.WEB_SEARCH: lambda p: p.get("query", ""),
    ToolName.SEARCH_CONFLUENCE: lambda p: p.get("query", ""),
}


def summarize_params(name: str, params: dict[str, Any]) -> str:
    """Return a one-line preview of a tool call's params for the CLI display."""
    summarizer = SUMMARIZERS.get(name)
    return summarizer(params) if summarizer else str(params)[:100]


def tool_header_label(name: str, params: dict[str, Any]) -> str:
    """Return the label half of a ``tool → {name}: {label}`` header.

    Falls back to the model's ``reason`` when present, otherwise to
    the param summary, otherwise to empty (in which case the caller
    should render just ``tool → {name}`` with no trailing colon).

    Single source of truth for all three CLI frontends — classic,
    rich, and TUI. Each CLI wraps the returned string in its own
    styling (ANSI, Rich ``Text``, Textual ``Static``) but the text
    content is identical across them.
    """
    return params.get("reason") or summarize_params(name, params)


# Tools whose ``execute()`` is never called — the processor intercepts
# them and routes their content through ``on_thinking`` / ``on_text``
# streaming instead. UI code skips the "running" spinner for these
# since the streamed content itself is the progress indicator.
# Derived from each tool's ``is_intercepted`` class attribute so the
# registry stays the single source of truth — adding a new intercepted
# tool means setting the flag on its class, not touching this list.
INTERCEPTED_TOOLS: frozenset[str] = frozenset(
    tool.name for tool in get_default_tools() if tool.is_intercepted
)
