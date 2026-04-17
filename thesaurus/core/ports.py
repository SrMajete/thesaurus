"""Hexagonal ports — protocols and data types at the core boundary.

Three ports define how the core communicates with the outside world:

1. **LLMClient** (here) — outbound port for LLM inference.
2. **AgentCallbacks** (in ``agent.py``) — inbound port for I/O rendering.
3. **Tool Protocol** (in ``tools/base.py``) — outbound port for tool execution.

``ToolCall`` and ``AgentResponse`` are the data types that cross the
core boundary. They are pure frozen dataclasses with zero infrastructure
dependencies, defined here so that core modules (``processor.py``,
``agent.py``) import from the ports module rather than from any adapter.
"""

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A single tool-use request extracted from the model's response."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class AgentResponse:
    """Parsed result of a single LLM API call.

    ``content`` holds the raw content blocks in dict form, ready to be
    appended directly to the messages list as the assistant's response.
    ``tool_calls`` is a parsed extraction for the agent loop.
    """

    content: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    @property
    def context_tokens(self) -> int:
        """Total context size: input + cache + output.

        Output is included because the assistant message becomes input
        on the next turn.
        """
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )


class LLMClient(Protocol):
    """Outbound port for LLM inference.

    The core calls this to get streaming model responses. The adapter
    (e.g. ``AnthropicLLMClient``) implements it with a specific SDK.
    The composition root injects the adapter at ``Agent`` construction.

    Streaming callbacks are part of the contract because the core
    (processor) needs to route deltas to the right display callback
    as they arrive. The adapter decides *when* to fire them; the core
    decides *what to do* with the deltas.
    """

    async def stream_response(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
        on_tool_start: Callable[[str, str, dict[str, Any]], None],
    ) -> AgentResponse: ...
