"""Run Python tool — executes Python code snippets in a fresh subprocess.

Dedicated channel for experimentation, hypothesis testing, and verification.
Each call spawns a new ``python -c <code>`` subprocess — stateless, no REPL,
no persistent globals between calls. Avoids shell escaping entirely by using
``create_subprocess_exec`` instead of ``create_subprocess_shell``.

The model should reach for this instead of bash's ``python -c "..."`` when
it wants to check how something behaves, compute a value, or verify a change.
"""

import asyncio
import sys
from typing import Any

from .base import ToolName, _format_subprocess_output


class RunPythonTool:
    """Execute a Python code snippet and return its combined stdout/stderr output."""

    name = ToolName.RUN_PYTHON
    description = (
        "Execute a Python code snippet in a fresh subprocess and return its output. "
        "Use this for experimentation, hypothesis testing, and verification — when "
        "you want to check how something behaves, compute a value, test a regex, "
        "explore a library, or verify that code you just wrote actually works. "
        "Each call is stateless (no persistent globals between calls), so re-import "
        "anything you need. Prefer this over bash's `python -c` for running Python."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "The Python code to execute. Use print() to produce output — "
                    "expression results are not auto-displayed. Runs with the "
                    "agent's working directory, so local imports work."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds. Defaults to 120.",
            },
        },
        "required": ["code"],
    }
    needs_permission = True
    is_parallelizable = True

    async def execute(self, *, code: str, timeout: int = 120) -> str:
        timeout = max(1, timeout)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: code timed out after {timeout} seconds."

        except OSError as e:
            return f"Error: {e}"

        return _format_subprocess_output(stdout, stderr, process.returncode)
