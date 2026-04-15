"""Run Python tool — executes Python code snippets in a fresh subprocess.

Each call spawns a new ``python -c <code>`` subprocess — stateless, no REPL,
no persistent globals between calls. Uses ``create_subprocess_exec`` to avoid
shell escaping issues.
"""

import sys
from typing import Any

from .base import ToolName
from ._subprocess import run_subprocess


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
    is_intercepted = False

    async def execute(self, *, code: str, timeout: int = 120) -> str:
        return await run_subprocess(
            sys.executable, "-c", code, timeout=max(1, min(timeout, 600))
        )
