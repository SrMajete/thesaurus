"""Tool registry — the single place that defines which tools are available.

To add a new tool: import it here and append an instance to the list.
"""

from typing import Awaitable, Callable

from .ask_user import AskUserTool
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


def get_default_tools(
    ask_user_question: Callable[[str], Awaitable[str]],
) -> list[Tool]:
    """Return instances of all built-in tools.

    ``ask_user_question`` is the callback the ``ask_user`` tool uses to
    prompt the user — typically ``TerminalOutput.ask_user_question`` in the
    CLI, but can be any async function that takes a question and returns
    a string answer.
    """
    return [
        MakePlanTool(),
        RunBashTool(),
        RunPythonTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobFilesTool(),
        GrepFilesTool(),
        AskUserTool(ask_user_question=ask_user_question),
        SendResponseTool(),
    ]
