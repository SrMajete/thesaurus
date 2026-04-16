"""Agent — owns all conversation state.

The Agent class holds the conversation messages, tools, and API client.
The single public method ``process_input()`` takes a user message and
delegates to the loop in ``processor.py`` for execution.

Decoupled from I/O via callbacks so the agent can be driven by a CLI,
web UI, test harness, or anything else.
"""

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

from .processor import run_loop
from .messages import user_message
from .prompts import environment_info
from .tools.base import Tool, tools_to_api_format


# All tool names must be snake_case with at least two words joined by
# underscores (e.g., ``read_file``, ``make_plan``, ``run_bash``). Enforced
# at Agent construction so any future tool added without following the
# convention fails loudly at startup instead of silently landing in the API.
_TOOL_NAME_PATTERN = re.compile(r"^[a-z]+(?:_[a-z]+)+$")


# ───────────────────────────────────────────────────────────────────────────
# Callback contract
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentCallbacks:
    """I/O callbacks that the agent delegates to the caller.

    This is the boundary between the agent's logic and the outside world.
    The agent never calls ``print()`` or ``input()`` directly — it uses
    these instead.

    Every tool-related callback carries ``tool_use_id`` as its first
    argument — the identifier assigned by the Claude API to each
    ``tool_use`` block. Consumers that need to pair ``on_tool_result``
    with its earlier ``on_tool_start`` / ``ask_permission`` must key
    on that ID: relying on tool ``name`` is incorrect when the model
    issues several parallel calls to the same tool (their names
    collide, IDs don't). The TUI uses the ID; the classic and Rich
    CLIs pair positionally (their output is sequential and transient)
    and explicitly ``del`` the parameter.

    Design note on the uniform signature: every consumer receives the
    ID, even the two that drop it. The alternative — split into two
    callback surfaces, one ID-aware — adds more complexity than it
    removes (two parallel event shapes, double wiring in the
    processor) for one marginal win. A single contract with one line
    of ``del`` per opt-out is clearer, so the contract encodes what's
    *available* to every consumer, not what each one happens to need.
    """

    on_thinking: Callable[[str], None]
    on_text: Callable[[str], None]
    on_tool_start: Callable[[str, str, dict[str, Any]], None]
    on_tool_result: Callable[[str, str, dict[str, Any], str, bool], None]
    ask_permission: Callable[[str, str, dict[str, Any]], Awaitable[bool]]


# ───────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────


class Agent:
    """Owns all conversation state. Delegates loop logic to ``processor.py``.

    Create one instance per session. Call ``process_input()`` for each
    user message. The agent manages state; the loop manages execution.
    """

    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
        tools: list[Tool],
        callbacks: AgentCallbacks,
        max_turns: int = 10,
        max_context_tokens: int = 200_000,
        prune_context_threshold: float = 0.8,
    ) -> None:
        for tool in tools:
            if not _TOOL_NAME_PATTERN.match(tool.name):
                raise ValueError(
                    f"Tool name {tool.name!r} must be snake_case with ≥2 "
                    "words joined by underscores (e.g. 'read_file')."
                )

        self.client = client
        self.model = model
        self.tools = tools
        self.callbacks = callbacks
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self.prune_context_threshold = prune_context_threshold
        self.messages: list[dict[str, Any]] = []
        self.api_tools = tools_to_api_format(tools)
        # Cache the environment section once per session. It's injected into
        # the system prompt every turn, but the underlying data (cwd, git
        # branch/status, platform, shell) doesn't change within a session —
        # and the git calls are subprocess spawns we don't want to repeat.
        self.env_info: str = environment_info()
        # Current plan (thinking + roadmap) as markdown. Updated by the
        # processor every time the model calls the make_plan tool. Injected
        # into the system prompt each turn so the model sees the live plan
        # state when deciding the next step. None until the first make_plan
        # call.
        self.plan: str | None = None
        # Cumulative token counts across the session. Bumped by the
        # processor after every ``stream_response`` with the usage
        # returned by the API. ``total_input_tokens`` sums fresh input
        # + cache writes + cache reads — all tokens the model
        # processed. ``total_cached_input_tokens`` is the cached portion
        # (cache writes + cache reads) so UIs can show cache efficiency
        # without storing the per-category breakdown. Consumers (e.g.
        # the TUI counter and per-turn separator) read these directly.
        self.total_input_tokens: int = 0
        self.total_cached_input_tokens: int = 0
        self.total_output_tokens: int = 0
        # Context size (input + cache + output tokens) from the most
        # recent API response. Set by the processor every turn. Used by
        # the TUI for the context percentage display and by the processor
        # for the pruning threshold check.
        self.last_context_tokens: int = 0

    async def process_input(self, user_input: str) -> None:
        """Process a user message and run the agentic loop.

        Appends the user message, then delegates to the loop which
        handles API calls, tool execution, and message updates.
        """
        self.messages.append(user_message(user_input))
        await run_loop(self)
