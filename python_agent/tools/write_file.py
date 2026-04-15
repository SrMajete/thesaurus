"""Write file tool — creates or overwrites files.

Writes content to a file, creating parent directories if they don't exist.
"""

from pathlib import Path
from typing import Any

from .base import ToolName


class WriteFileTool:
    """Write content to a file, creating parent directories as needed."""

    name = ToolName.WRITE_FILE
    description = (
        "Write content to a file. Creates the file if it doesn't exist, or "
        "overwrites it if it does. Parent directories are created automatically. "
        "Use this instead of echo/cat heredocs via bash. "
        "Use edit_file for modifying existing files."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    }
    needs_permission = True
    is_parallelizable = False
    is_intercepted = False

    async def execute(self, *, file_path: str, content: str) -> str:
        path = Path(file_path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        
        except OSError as e:
            return f"Error: {e}"

        byte_count = len(content.encode())
        return f"Successfully wrote {byte_count} bytes to {file_path}"
