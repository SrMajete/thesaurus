"""Run-bash tool — executes shell commands and returns their output."""

from typing import Any

from .base import ToolName
from ._subprocess import run_subprocess


class RunBashTool:
    """Execute a shell command and return its combined stdout/stderr output."""

    name = ToolName.RUN_BASH
    description = (
        "Run a shell command and return its output. Use this for system commands, "
        "running scripts, installing packages, git operations, and any task that "
        "requires shell access. Prefer the dedicated read_file/write_file tools "
        "for file operations instead of cat/echo."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds. Defaults to 120.",
            },
        },
        "required": ["command"],
    }
    needs_permission = True
    is_parallelizable = True

    async def execute(self, *, command: str, timeout: int = 120) -> str:
        return await run_subprocess(command, timeout=max(1, timeout), shell=True)
