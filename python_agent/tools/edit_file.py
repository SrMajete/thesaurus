"""Edit file tool — surgical search-and-replace for existing files.

Finds an exact match of ``old_text`` in the target file and replaces it
with ``new_text``. The match must be unique — zero or multiple matches
are rejected with a clear error so the model can add context to
disambiguate.
"""

from pathlib import Path
from typing import Any

from .base import ToolName


class EditFileTool:
    """Apply a single search-and-replace edit to an existing file."""

    name = ToolName.EDIT_FILE
    description = (
        "Apply a search-and-replace edit to an existing file. Provide the exact "
        "text to find (old_text) and its replacement (new_text). old_text must "
        "match exactly once in the file — include enough surrounding context to "
        "make it unique. Use write_file for creating new files."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file to edit.",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "The exact text to find in the file. Must match exactly once. "
                    "Include enough context (surrounding lines) to be unique."
                ),
            },
            "new_text": {
                "type": "string",
                "description": (
                    "The replacement text. Use an empty string to delete old_text."
                ),
            },
        },
        "required": ["file_path", "old_text", "new_text"],
    }
    needs_permission = True
    is_parallelizable = False

    _MAX_FILE_SIZE = 1_000_000_000  # 1 GB

    async def execute(
        self, *, file_path: str, old_text: str, new_text: str
    ) -> str:
        if not old_text:
            return "Error: old_text must not be empty."

        if old_text == new_text:
            return "Error: old_text and new_text are identical — nothing to change."

        path = Path(file_path)

        if not path.exists():
            return f"Error: file not found: {file_path}"

        if not path.is_file():
            return f"Error: not a file: {file_path}"

        if path.stat().st_size > self._MAX_FILE_SIZE:
            return f"Error: file exceeds 1 GB limit: {file_path}"

        try:
            content = path.read_bytes()
        except PermissionError:
            return f"Error: permission denied: {file_path}"

        if b"\x00" in content[:8192]:
            return f"Error: {file_path} appears to be a binary file."

        text = content.decode(errors="replace")
        count = text.count(old_text)

        if count == 0:
            return "Error: old_text not found in file."

        if count > 1:
            return (
                f"Error: old_text found {count} times — must be unique. "
                "Add surrounding context to disambiguate."
            )

        match_start = text.index(old_text)
        start_line = text[:match_start].count("\n") + 1
        new_text_lines = new_text.count("\n") + 1 if new_text else 0
        end_line = start_line + new_text_lines - 1

        new_content = text.replace(old_text, new_text, 1)

        try:
            path.write_text(new_content)
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as e:
            return f"Error: {e}"

        if new_text:
            return f"Applied edit to {file_path} (lines {start_line}-{end_line})"
        return f"Applied edit to {file_path} (deleted from line {start_line})"
