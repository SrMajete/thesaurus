"""Tool registry — the single place that defines which tools are available.

To add a new tool: import it here and append an instance to the list.
"""

from .base import Tool
from .edit_file import EditFileTool
from .glob_files import GlobFilesTool
from .grep_files import GrepFilesTool
from .make_plan import MakePlanTool
from .read_file import ReadFileTool
from .run_bash import RunBashTool
from .run_python import RunPythonTool
from .send_response import SendResponseTool
from .write_file import WriteFileTool


def get_default_tools() -> list[Tool]:
    """Return instances of all built-in tools."""
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
