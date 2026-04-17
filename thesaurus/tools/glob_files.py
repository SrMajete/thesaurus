"""Glob-files tool — find files by name pattern.

Uses ``pathlib.Path.glob()`` to match files in a directory tree and
returns paths sorted by modification time (newest first).  No subprocess
overhead, no external dependencies.
"""

from pathlib import Path
from typing import Any

from .base import ToolName

_MAX_RESULTS = 200


class GlobFilesTool:
    """Find files whose names match a glob pattern."""

    name = ToolName.GLOB_FILES
    description = (
        "Find files by name pattern using glob syntax (e.g., '**/*.py', "
        "'src/**/*.ts'). Returns matching paths sorted by modification time. "
        "Use this instead of find or ls via bash."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '**/*.py', 'src/*.ts').",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search in. Defaults to the current "
                    "working directory."
                ),
            },
        },
        "required": ["pattern"],
    }
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    async def execute(self, *, pattern: str, path: str = ".") -> str:
        search_dir = Path(path)

        if not search_dir.is_dir():
            return f"Error: not a directory: {path}"

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matches = sorted(
            (p for p in search_dir.glob(pattern) if p.is_file()),
            key=_mtime,
            reverse=True,
        )

        total = len(matches)

        if total == 0:
            return f"No files matching '{pattern}' in {path}"

        result = "\n".join(str(p) for p in matches[:_MAX_RESULTS])

        if total > _MAX_RESULTS:
            result += (
                f"\n\n({total} total, showing first {_MAX_RESULTS}. "
                "Use a more specific pattern.)"
            )

        return result
