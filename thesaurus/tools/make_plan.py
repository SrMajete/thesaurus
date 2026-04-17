"""Make-plan tool — forces turn-level strategic reasoning.

Since LLMs are autoregressive, writing a plan before other tool calls
naturally influences and improves the quality of those calls. The plan
text shapes the model's decisions for the rest of the turn.
"""

from typing import Any

from .base import ToolName


class MakePlanTool:
    """Produce a structured plan before executing other tools.

    Stateful via the processor: when intercepted, the processor writes the
    combined thinking + roadmap to ``agent.plan``, which is then injected
    into the next turn's system prompt as dynamic context. This gives the
    model a live view of its own plan without re-writing it from memory.
    """

    name = ToolName.MAKE_PLAN
    description = "Produce a structured plan for the task. Call before any other tool."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
                "description": (
                    "Your reasoning for this turn: what you learned from the "
                    "previous results, what changed, what it means for the "
                    "roadmap, and why you're doing what you're doing next. "
                    "Free-form markdown."
                ),
            },
            "roadmap": {
                "type": "string",
                "description": (
                    "The full task decomposition as a numbered markdown list, "
                    "from the current state to task completion. Each item "
                    "follows the format: "
                    "'N. [status] description — tool_call(concrete_args)'. "
                    "Status is one of: pending, in_progress, completed, "
                    "dropped. Exactly one item may be in_progress at a time. "
                    "Keep completed and dropped items visible (don't delete "
                    "them). Every item names exactly one concrete tool call "
                    "with real arguments; for args that depend on earlier "
                    "steps, write 'TBD after step N' and concretize before "
                    "executing. The final item is always 'send_response'."
                ),
            },
        },
        "required": ["thinking", "roadmap"],
    }
    # is_intercepted=True signals that ``execute()`` is never called —
    # the processor handles plan dispatch (updating ``agent.plan``) and
    # the LLM adapter streams ``thinking`` / ``roadmap`` fields live to
    # the user. The other two flags are still required by the Protocol;
    # the honest semantic values are: needs_permission=False (we never
    # prompt for plans), is_parallelizable=False (plans mutate
    # ``agent.plan`` and cannot race).
    needs_permission = False
    is_parallelizable = False
    is_intercepted = True

    async def execute(self, *, thinking: str, roadmap: str) -> str:
        """Protocol stub. Make-plan handling is intercepted by the
        processor before tool execution; this method is never called in
        the normal flow. Present only to satisfy the ``Tool`` Protocol.
        """
        return "Plan acknowledged."
