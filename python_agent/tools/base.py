"""Tool protocol and helper functions.

Defines the contract every tool must implement and provides utilities for
looking up tools and converting them to the Anthropic API format. Concrete
tools satisfy the protocol with plain class attributes and one async method —
no inheritance or decorators needed.
"""

import copy
from enum import StrEnum
from typing import Any, Protocol


class ToolName(StrEnum):
    """Canonical tool names. Single source of truth for dispatch.

    Every tool sets ``name = ToolName.X`` instead of a raw string literal,
    and every consumer (processor, api_client, cli) imports ``ToolName``
    and compares against ``ToolName.X`` instead of ``"x"``. This collapses
    the raw-string dispatch keys scattered across the consumer files down
    to one authoritative list and makes future renames a one-line change.

    ``StrEnum`` (Python 3.11+) is a first-class subclass of ``str``, so
    ``ToolName.RUN_BASH`` IS the string ``"run_bash"`` at runtime — no
    ``.value`` lookups anywhere. The ``Tool`` Protocol field ``name: str``
    accepts ``ToolName.X`` directly. Consumers can use
    ``tool.name == ToolName.RUN_BASH``, ``dict[ToolName.X] = ...``, and
    f-string interpolation without any explicit conversions.

    When adding a new tool: add one entry here, use it as the tool's
    ``name`` attribute, and add matching branches in any consumer that
    dispatches on tool name. The regex assertion in ``Agent.__init__``
    enforces the verb_noun naming convention at construction time;
    ``ToolName`` enforces the single-source-of-truth contract.
    """
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    GLOB_FILES = "glob_files"
    GREP_FILES = "grep_files"
    RUN_BASH = "run_bash"
    RUN_PYTHON = "run_python"
    MAKE_PLAN = "make_plan"
    SEND_RESPONSE = "send_response"
    SPAWN_AGENT = "spawn_agent"


REASON_FIELD_DESCRIPTION = (
    "A short one-sentence justification for WHY you are calling this tool "
    "right now — written for the user watching in the terminal. Focus on the "
    "purpose / trigger, NOT a preview of the action or content. The user will "
    "see the action and any output separately. "
    "Good: 'Need to understand exports before refactoring the package.' "
    "Good: 'Verifying the auth fix did not break login.' "
    "Good: 'Investigation complete, ready to share findings.' "
    "Bad: 'Reading __init__.py.' (describes the action, not why). "
    "Bad: 'Exploring the project structure and key files.' (previews content, "
    "no reason). Keep it under 20 words."
)


class Tool(Protocol):
    """Contract that every tool must satisfy.

    Uses plain attribute annotations so concrete tools can just set
    class-level constants. Three flags control dispatch and lifecycle:

    - ``needs_permission``: if True, the user must approve the call before
      it executes. Read-oriented tools (``read_file``, ``make_plan``) set
      False; anything that touches state, spawns a subprocess, or talks
      to external systems sets True.
    - ``is_parallelizable``: if True, the tool is safe to run concurrently
      with peers in the same batch via ``asyncio.gather``. Tools that do
      blocking synchronous I/O or that share mutable state with each
      other must set False.
    - ``is_intercepted``: if True, ``execute()`` is never called — the
      processor handles the tool specially (``make_plan`` updates
      ``agent.plan``; ``send_response`` ends the turn) and ``api_client``
      streams the tool's input fields live to the user via
      ``on_thinking`` / ``on_text``. Default: False (most tools execute
      normally). UI code reads this flag to skip per-tool spinners since
      the streamed content itself is the progress indicator.

    The first two flags must be declared explicitly — no defaults. A
    new tool that forgets to think about either dimension breaks at
    import time, not at runtime. All four combinations are meaningful:

    - (False, True)  → auto-parallel        (read_file, glob_files, grep_files)
    - (False, False) → auto-serial          (make_plan, send_response)
    - (True,  True)  → approve-then-parallel (run_bash, run_python)
    - (True,  False) → approve-then-serial  (write_file, edit_file)
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    needs_permission: bool
    is_parallelizable: bool
    is_intercepted: bool = False

    async def execute(self, **params: Any) -> str:
        """Run the tool and return a string result.

        Parameters are unpacked from the LLM's tool_use input dict. Any
        exception raised here is caught by the agent loop, converted to an
        error string, and sent back to the LLM so it can recover.

        Intercepted tools (``is_intercepted=True``) never have this
        called — the processor handles them specially.
        """
        ...


def find_tool(name: str, tools: list[Tool]) -> Tool | None:
    """Look up a tool by name. Returns None if not found."""
    for tool in tools:
        if tool.name == name:
            return tool

    return None


def _api_format(tool: Tool) -> dict[str, Any]:
    """Format a single tool for the Anthropic API, with ``reason`` injected
    as the first property and first required field.
    """
    schema = copy.deepcopy(tool.input_schema)
    schema["properties"] = {
        "reason": {"type": "string", "description": REASON_FIELD_DESCRIPTION},
        **schema.get("properties", {}),
    }
    existing_required = schema.get("required", [])
    if "reason" not in existing_required:
        schema["required"] = ["reason", *existing_required]
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


def tools_to_api_format(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert tools to the Anthropic API ``tools`` parameter format.

    A ``reason`` field is auto-injected as the *first* property of every
    tool's schema so the model writes it before the other fields. This
    matters for streamable tools (make_plan, send_response): the streaming
    renderer in ``api_client.py`` fires ``on_tool_start`` as soon as the
    first field has content, and placing ``reason`` first guarantees the
    header line reads the reason instead of a fallback summary from partial
    content.

    The last tool in the returned list carries ``cache_control`` so the
    Anthropic API caches the full tool-definition block across turns.
    """
    formatted = [_api_format(tool) for tool in tools]
    if formatted:
        formatted[-1]["cache_control"] = {"type": "ephemeral"}
    return formatted
