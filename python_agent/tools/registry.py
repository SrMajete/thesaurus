"""Tool registry — the single place that defines which tools are available.

To add a new tool: import it here and append an instance to the list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool
from .edit_file import EditFileTool
from .glob_files import GlobFilesTool
from .grep_files import GrepFilesTool
from .make_plan import MakePlanTool
from .read_file import ReadFileTool
from .run_bash import RunBashTool
from .run_python import RunPythonTool
from .send_response import SendResponseTool
from .spawn_agent import SpawnAgentTool
from .write_file import WriteFileTool

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from ..agent import AgentCallbacks


def get_default_tools() -> list[Tool]:
    """Return instances of all built-in tools (no external dependencies)."""
    return [
        MakePlanTool(),
        RunBashTool(),
        RunPythonTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobFilesTool(),
        GrepFilesTool(),
        SendResponseTool(),
    ]


def get_all_tools(
    client: AsyncAnthropic | AsyncAnthropicBedrock,
    model: str,
    max_turns: int,
    parent_callbacks: AgentCallbacks,
) -> list[Tool]:
    """Full tool set including tools that need runtime dependencies."""
    base = get_default_tools()
    return [*base, SpawnAgentTool(client, model, max_turns, base, parent_callbacks)]
