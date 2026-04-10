"""Tools package — protocol, built-in tools, and registry.

Public API: ``Tool`` (the Protocol), ``find_tool`` (lookup helper),
``tools_to_api_format`` (convert to Anthropic API format), and
``get_default_tools`` (registry constructor). The concrete tool classes
(RunBashTool, ReadFileTool, etc.) are internal — accessible via their
submodules if needed but not re-exported here.
"""

from .base import Tool, find_tool, tools_to_api_format
from .registry import get_default_tools

__all__ = [
    "Tool",
    "find_tool",
    "tools_to_api_format",
    "get_default_tools",
]
