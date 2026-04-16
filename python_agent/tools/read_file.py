"""Read file tool — reads file contents with line numbers.

Returns file content formatted with line numbers for easy reference,
with support for partial reads via offset and limit parameters.
"""

from pathlib import Path
from typing import Any

from ._helpers import is_binary, validate_file_path
from .base import ToolName

DEFAULT_LIMIT = 2000


class ReadFileTool:
    """Read a file and return its contents with line numbers."""

    name = ToolName.READ_FILE
    description = (
        "Read a file from the filesystem and return its contents with line numbers. "
        "Supports partial reads via offset (start line) and limit (max lines). "
        "Use this instead of cat/head/tail via bash."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based). Defaults to 0.",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of lines to read. Defaults to {DEFAULT_LIMIT}.",
            },
        },
        "required": ["file_path"],
    }
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    async def execute(
        self, *, file_path: str, offset: int = 0, limit: int = DEFAULT_LIMIT
    ) -> str:
        path = Path(file_path)

        err = validate_file_path(path)
        if err:
            return err

        try:
            content = path.read_bytes()
        except PermissionError:
            return f"Error: permission denied: {file_path}"

        if is_binary(content):
            return f"Error: {file_path} appears to be a binary file."

        text = content.decode(errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)

        offset = max(0, offset)
        limit = max(0, limit)
        selected = lines[offset : offset + limit]
        numbered = [
            f"{i + offset + 1:>4}\t{line}" for i, line in enumerate(selected)
        ]
        result = "\n".join(numbered)

        if offset + limit < total_lines:
            result += f"\n\n... ({total_lines - offset - limit} more lines)"

        return result
