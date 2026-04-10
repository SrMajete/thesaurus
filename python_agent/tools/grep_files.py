"""Grep-files tool — search file contents by regex pattern.

Calls ``ripgrep`` (``rg``) as a subprocess for fast, recursive content
search.  Returns matching lines in ``file:line:content`` format, capped
at a configurable line limit to protect the context window.
"""

import asyncio
from typing import Any

from .base import ToolName

_MAX_LINES = 200
_TIMEOUT = 30


class GrepFilesTool:
    """Search file contents for a regex pattern via ripgrep."""

    name = ToolName.GREP_FILES
    description = (
        "Search file contents for a regex pattern using ripgrep. Returns "
        "matching lines with file paths and line numbers. "
        "Use this instead of grep or rg via bash."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for in file contents.",
            },
            "path": {
                "type": "string",
                "description": (
                    "File or directory to search. Defaults to the current "
                    "working directory."
                ),
            },
            "include": {
                "type": "string",
                "description": (
                    "Glob pattern to filter files (e.g., '*.py', '*.{ts,tsx}')."
                ),
            },
        },
        "required": ["pattern"],
    }
    needs_permission = False
    is_parallelizable = True

    async def execute(
        self, *, pattern: str, path: str = ".", include: str = ""
    ) -> str:
        args = [
            "rg", "--color=never", "--no-heading",
            "--line-number", "--max-columns=200",
        ]

        for vcs_dir in (".git", ".svn", ".hg"):
            args.extend(["--glob", f"!{vcs_dir}"])

        if include:
            args.extend(["--glob", include])

        # -e prevents patterns starting with '-' from being parsed as flags
        args.extend(["-e", pattern, path])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_TIMEOUT,
            )
        except FileNotFoundError:
            return "Error: ripgrep (rg) not found. Install: pip install ripgrep"
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: search timed out after {_TIMEOUT} seconds."

        if process.returncode == 1:
            return f"No matches for '{pattern}' in {path}"

        if process.returncode not in (0, 1):
            error = stderr.decode(errors="replace").strip()
            return f"Error: {error}" if error else "Error: search failed."

        output = stdout.decode(errors="replace")
        lines = output.splitlines()

        if len(lines) > _MAX_LINES:
            result = "\n".join(lines[:_MAX_LINES])
            result += (
                f"\n\n({len(lines)} total matches, "
                f"showing first {_MAX_LINES}.)"
            )
            return result

        return output
