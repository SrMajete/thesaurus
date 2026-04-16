"""Spawn sub-agent tool — delegates focused tasks to child agents.

Creates a new ``Agent`` + ``Processor`` pair with a restricted tool set,
a role-specific system prompt, and auto-approved permissions. The child
runs its full agent loop, and the result (last plan + response) is
returned to the parent as the tool output.

Callback forwarding
-------------------
The child's ``on_tool_start`` and ``on_tool_result`` are forwarded to the
parent's TUI so the user sees live activity. Three dispatch paths are
handled:

- **auto** (read_file, glob, grep): processor calls ``on_tool_start``
  in ``_execute_decisions`` → forwarded via ``_filtered_tool_start``.
- **approve** (run_bash, run_python): processor calls ``ask_permission``
  → ``_approve_with_header`` fires ``on_tool_start`` and returns True.
- **intercepted** (make_plan, send_response): ``stream_response`` fires
  ``on_tool_start`` → ``_filtered_tool_start`` blocks it (no empty
  header widgets in the parent TUI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent_types import AGENT_TYPES, get_agent_type
from .base import ToolName

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from ..agent import Agent, AgentCallbacks

# Tools whose on_tool_start from stream_response should NOT be forwarded
# to the parent TUI — their content (thinking, response) is suppressed,
# so the header widget would appear empty.
# Defined here (not imported from tool_summaries) to avoid a circular
# import: tool_summaries → tools/__init__ → registry → spawn_agent.
_INTERCEPTED_NAMES = frozenset({ToolName.MAKE_PLAN, ToolName.SEND_RESPONSE})


def _extract_result(messages: list[dict[str, Any]]) -> str:
    """Extract the child's last plan and response from its message history.

    Walks forward through assistant messages to find the last make_plan
    and send_response tool_use blocks. Last assignment wins — simpler
    than reverse search.
    """
    last_plan = ""
    last_response = ""

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") != "tool_use":
                continue
            if block["name"] == ToolName.MAKE_PLAN:
                thinking = block["input"].get("thinking", "")
                roadmap = block["input"].get("roadmap", "")
                last_plan = f"### Thinking\n{thinking}\n\n### Roadmap\n{roadmap}"
            elif block["name"] == ToolName.SEND_RESPONSE:
                last_response = block["input"].get("response", "")

    if not last_response:
        return (
            f"Sub-agent did not complete (hit max_turns).\n\n"
            f"## Last Plan\n{last_plan}"
        )

    return f"## Plan\n{last_plan}\n\n## Response\n{last_response}"


class SpawnAgentTool:
    """Launch a read-only sub-agent to handle a focused task."""

    name = ToolName.SPAWN_AGENT
    description = (
        "Launch a read-only sub-agent to handle a focused task. Each agent "
        "type has specific capabilities and tools. Use this when the task "
        "benefits from a specialized perspective (code review, bug hunting) "
        "or when you want to run focused work without consuming your own "
        "context window. Do NOT use for simple file reads or searches you "
        "can do directly."
    )
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    input_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A short (3-5 word) description of the task.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "The task for the sub-agent to perform. Brief it like a "
                    "smart colleague who just walked into the room — it hasn't "
                    "seen this conversation, doesn't know what you've tried, "
                    "doesn't understand why this task matters. Explain what "
                    "you're trying to accomplish, describe what you've already "
                    "learned or ruled out, and give enough context that the "
                    "agent can make judgment calls."
                ),
            },
            "agent_type": {
                "type": "string",
                "enum": list(AGENT_TYPES.keys()),
                "description": (
                    "code_reviewer: quality, patterns, architecture review. "
                    "bug_hunter: hypothesis-driven bug finding with test execution."
                ),
            },
        },
        "required": ["description", "prompt", "agent_type"],
    }

    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
        max_turns: int,
        base_tools: list,
        parent_callbacks: AgentCallbacks,
    ) -> None:
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.base_tools = base_tools
        self.parent_callbacks = parent_callbacks
        self.parent_agent: Agent | None = None  # set post-construction

    async def execute(
        self, *, description: str, prompt: str, agent_type: str
    ) -> str:
        """Spawn a child agent, run it to completion, return the result."""
        del description  # used by summarizer, not by execution

        # Lazy import to break circular dependency:
        # spawn_agent → agent → processor → api_client → tools → registry → spawn_agent
        from ..agent import Agent, AgentCallbacks  # noqa: F811

        assert self.parent_agent is not None, (
            "SpawnAgentTool.parent_agent must be set before use"
        )

        agent_def = get_agent_type(agent_type)

        # Build child tool list from base_tools — single source of truth.
        allowed = set(agent_def.tools) | {ToolName.MAKE_PLAN, ToolName.SEND_RESPONSE}
        child_tools = [t for t in self.base_tools if t.name in allowed]

        # Forwarding callbacks: child tool activity appears live in
        # the parent TUI. See module docstring for dispatch path details.
        parent_start = self.parent_callbacks.on_tool_start

        async def _approve_with_header(
            tool_use_id: str, name: str, params: dict[str, Any]
        ) -> bool:
            """Auto-approve and register the tool header in parent TUI.

            The approve-path in the processor skips ``on_tool_start``, so
            ``ask_permission`` must fire it — otherwise ``on_tool_result``
            will assert-fail on an unregistered ``tool_use_id``.
            """
            parent_start(tool_use_id, name, params)
            return True

        def _filtered_tool_start(
            tool_use_id: str, name: str, params: dict[str, Any]
        ) -> None:
            """Forward ``on_tool_start`` only for non-intercepted tools.

            ``stream_response`` fires ``on_tool_start`` for make_plan and
            send_response during streaming — without this filter, those
            would create empty header widgets in the parent TUI (content
            is suppressed via the no-op ``on_thinking``/``on_text``).
            """
            if name not in _INTERCEPTED_NAMES:
                parent_start(tool_use_id, name, params)

        child_callbacks = AgentCallbacks(
            on_thinking=lambda _: None,
            on_text=lambda _: None,
            on_tool_start=_filtered_tool_start,
            on_tool_result=self.parent_callbacks.on_tool_result,
            ask_permission=_approve_with_header,
        )

        child = Agent(
            client=self.client,
            model=self.model,
            tools=child_tools,
            callbacks=child_callbacks,
            max_turns=self.max_turns,
            prompt_blocks=agent_def.prompt_blocks,
            system_preamble=agent_def.system_preamble,
        )

        try:
            await child.process_input(prompt)
        finally:
            # Propagate even on failure — API calls still consumed tokens.
            self.parent_agent.accumulate_tokens(
                child.total_input_tokens,
                child.total_cached_input_tokens,
                child.total_output_tokens,
            )

        return _extract_result(child.messages)
