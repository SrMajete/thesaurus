"""Ask user tool — allows the model to ask clarification questions.

Instead of guessing when instructions are ambiguous, the model can
explicitly pause and ask the user a question. The user's answer is
returned as the tool result.

The actual I/O is delegated to a callback injected at construction
(same pattern as ``AgentCallbacks``), so the tool works with any
frontend — terminal, web UI, test harness — not just the CLI.
"""

from typing import Any, Awaitable, Callable

from .base import ToolName


class AskUserTool:
    """Ask the user a question to gather information or clarify ambiguity."""

    name = ToolName.ASK_USER
    description = (
        "Ask the user a question to gather information, clarify ambiguous "
        "instructions, understand preferences, or get a decision. Use this "
        "instead of guessing when the user's intent is unclear."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask. Should be clear and specific.",
            },
        },
        "required": ["question"],
    }
    # No permission prompt — the call IS the user interaction, so gating
    # it behind a Y/n prompt would produce a double-prompt (permit, then
    # answer). NOT parallelizable: input() blocks the event loop and would
    # wedge any concurrent ask_user peers in the same batch.
    needs_permission = False
    is_parallelizable = False

    def __init__(self, ask_user_question: Callable[[str], Awaitable[str]]) -> None:
        self._ask = ask_user_question

    async def execute(self, *, question: str) -> str:
        return await self._ask(question)
