"""Tool registry — the single place that defines which tools are available.

To add a new tool: import it here and append an instance to the list.
"""

from .base import Tool
from .edit_file import EditFileTool
from .fetch_url import FetchUrlTool
from .glob_files import GlobFilesTool
from .grep_files import GrepFilesTool
from .make_plan import MakePlanTool
from .read_file import ReadFileTool
from .run_bash import RunBashTool
from .run_python import RunPythonTool
from .send_response import SendResponseTool
from .web_search import WebSearchTool
from .write_file import WriteFileTool


def get_default_tools(
    *,
    confluence_url: str | None = None,
    confluence_email: str | None = None,
    confluence_api_key: str | None = None,
) -> list[Tool]:
    """Return instances of all built-in tools.

    Optional tools (like Confluence search) are only included when
    their required configuration is fully provided.
    """
    tools: list[Tool] = [
        MakePlanTool(),
        RunBashTool(),
        RunPythonTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobFilesTool(),
        GrepFilesTool(),
        FetchUrlTool(),
        WebSearchTool(),
        SendResponseTool(),
    ]
    if confluence_url and confluence_email and confluence_api_key:
        from .search_confluence import SearchConfluenceTool
        tools.append(SearchConfluenceTool(
            url=confluence_url,
            email=confluence_email,
            api_key=confluence_api_key,
        ))
    return tools
