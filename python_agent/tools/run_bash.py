"""Run-bash tool — executes shell commands and returns their output.

Runs commands via asyncio subprocess, captures both stdout and stderr,
and enforces a configurable timeout to prevent runaway processes.
"""

import asyncio
from typing import Any

from .base import ToolName, _format_subprocess_output


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
        timeout = max(1, timeout)
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: command timed out after {timeout} seconds."
        
        except OSError as e:
            return f"Error: {e}"

        return _format_subprocess_output(stdout, stderr, process.returncode)
