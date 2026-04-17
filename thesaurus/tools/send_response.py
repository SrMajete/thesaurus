"""Send-response tool — forces the model to reason before every user response.

By routing all user-facing responses through a tool, the model must
produce structured reasoning before every response, not just before
tool calls. The response text is returned as the tool result.
"""

from typing import Any

from .base import ToolName


class SendResponseTool:
    """Send a response to the user."""

    name = ToolName.SEND_RESPONSE
    description = (
        "Use this tool to send your final response to the user. "
        "Always call make_plan before calling send_response. "
        "Only call send_response when every step in your roadmap is [completed] or "
        "explicitly dropped with a justification in the plan."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The text response to show to the user.",
            },
        },
        "required": ["response"],
    }
    # is_intercepted=True signals that ``execute()`` is never called —
    # the processor handles send_response by ending the turn and
    # the LLM adapter streams the ``response`` field live to the user.
    # The honest semantic values for the other two flags:
    # needs_permission=False (the response IS the user interaction —
    # prompting "may I send this?" is nonsensical), is_parallelizable=False
    # (there's only ever one response per turn; run_loop returns on the
    # first send_response call).
    needs_permission = False
    is_parallelizable = False
    is_intercepted = True

    async def execute(self, *, response: str) -> str:
        """Protocol stub. Send-response handling is intercepted by the
        processor before tool execution; this method is never called in
        the normal flow. Present only to satisfy the ``Tool`` Protocol.
        """
        return response
