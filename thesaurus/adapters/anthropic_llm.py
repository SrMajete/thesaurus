"""AnthropicLLMClient — ``LLMClient`` adapter wrapping the Anthropic SDK.

Implements the ``LLMClient`` port defined in ``core.ports`` using the
Anthropic Python SDK for streaming inference. All SDK-specific details
(streaming event types, partial JSON parsing with jiter, cache control,
tool input streaming) are encapsulated here.

Tool input streaming
--------------------
For tools listed in ``_STREAM_FIELDS_BY_TOOL`` (currently ``send_response``
and ``make_plan``), the named input fields are streamed character-by-character
to the user via the appropriate callback as the model generates them — even
though the content lives inside a tool call. This works by accumulating each
``input_json_delta``'s ``partial_json`` bytes and re-parsing the buffer with
``jiter`` in ``trailing-strings`` mode, which emits in-progress strings
before they close. New characters of each tracked field are forwarded as
they appear.
"""

import json
import logging
import time
from typing import Any, Callable

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, APIError
from jiter import from_json as _parse_partial_json

from thesaurus.core.ports import AgentResponse, ToolCall
from thesaurus.tools.base import ToolName

logger = logging.getLogger(__name__)


# Map of tool name → ordered list of input fields to stream live to the user.
# Field order must match the order the model writes them in the JSON, which
# follows the input_schema field order. For make_plan, thinking precedes
# roadmap because that's how the schema is defined.
_STREAM_FIELDS_BY_TOOL: dict[str, list[str]] = {
    ToolName.MAKE_PLAN: ["thinking", "roadmap"],
    ToolName.SEND_RESPONSE: ["response"],
}

# Section header prepended to a field's rendered output the first time it
# is streamed. Fields not in this dict render without a prefix (e.g.
# ``send_response.response`` — the response text is its own content).
_FIELD_PREFIX: dict[str, str] = {
    "thinking": "## Thinking\n\n",
    "roadmap": "\n\n## Roadmap\n\n",
}


def _render_field(field: str, value: str) -> str:
    """Return the full rendered output for a streaming field's value so far.

    Strips leading whitespace from ``value`` so a stray leading newline
    inside the model's output doesn't stack with the prefix's trailing
    ``\\n\\n``. Prepends the field's section header if one is defined.
    Returns an empty string if the value is all whitespace — the caller
    should emit nothing and wait for more bytes.

    The caller tracks how many rendered chars it has already forwarded
    and emits only the suffix on each delta, so this function is safe to
    call repeatedly as ``value`` grows with each incoming delta.
    """
    stripped = value.lstrip()
    if not stripped:
        return ""
    return _FIELD_PREFIX.get(field, "") + stripped


# ───────────────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _with_message_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow-patched copy with cache_control on the last user
    message's last content block.  The original list is never mutated.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        content = msg["content"]
        if msg["role"] == "user" and isinstance(content, list) and content:
            last_block = {**content[-1], "cache_control": {"type": "ephemeral"}}
            patched_msg = {**msg, "content": content[:-1] + [last_block]}
            return messages[:i] + [patched_msg] + messages[i + 1:]
    return messages


class AnthropicLLMClient:
    """``LLMClient`` adapter wrapping the Anthropic Python SDK.

    Encapsulates the Anthropic-specific streaming, event parsing,
    partial JSON handling, and cache control. The core (processor)
    calls ``stream_response`` through the ``LLMClient`` protocol
    without knowing which SDK is behind it.
    """

    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    async def stream_response(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
        on_tool_start: Callable[[str, str, dict[str, Any]], None],
    ) -> AgentResponse:
        """Call the Claude API with streaming and return a parsed response.

        ``system`` is a list of ``TextBlockParam`` dicts, typically with
        ``cache_control`` on the static prefix block. See ``prompts.py``
        for the construction.

    For streamable tools (see ``_STREAM_FIELDS_BY_TOOL``), the tool header
    fires via ``on_tool_start`` as soon as the block starts producing content,
    and the field text streams live to ``on_text`` (respond) or ``on_thinking``
    (plan). The processor must skip its post-turn ``on_tool_start``/
    ``on_tool_result`` calls for these tools to avoid duplicate rendering.
    """
        logger.info(
            "API request: model=%s, messages=%d, tools=%s",
                self._model,
            len(messages),
            [t["name"] for t in tools],
        )

        start = time.monotonic()

        # Build API params — tool_choice "any" forces the model to always use a tool
        api_params: dict[str, Any] = {
            "model": self._model,
            "messages": _with_message_cache_breakpoint(messages),
            "system": system,
            "tools": tools,
            "tool_choice": {"type": "any"},
            "max_tokens": 16_384,
        }

        # Per-content-block streaming state. Reset on every content_block_start
        # for a streamable tool. Outside such blocks, ``stream_fields`` is None
        # and the input_json_delta branch is a no-op.
        #
        # ``streamed[field]`` tracks the number of *rendered* characters already
        # forwarded to the callback — not the number of source characters
        # consumed. The rendered output is ``_render_field(field, value)`` which
        # includes the section header prefix and strips leading whitespace, so
        # "has the section header been emitted?" is implicit in whether
        # ``streamed[field] > 0``.
        stream_fields: list[str] | None = None
        stream_cb: Callable[[str], None] | None = None
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        header_fired = False
        json_buf = b""
        streamed: dict[str, int] = {}

        try:
            async with self._client.messages.stream(**api_params) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            fields = _STREAM_FIELDS_BY_TOOL.get(block.name)
                            if fields:
                                stream_fields = fields
                                stream_cb = on_thinking if block.name == ToolName.MAKE_PLAN else on_text
                                current_tool_id = block.id
                                current_tool_name = block.name
                                header_fired = False
                                json_buf = b""
                                streamed = {f: 0 for f in fields}

                    elif event.type == "content_block_stop":
                        stream_fields = None
                        stream_cb = None
                        current_tool_id = None
                        current_tool_name = None

                    elif event.type == "content_block_delta":
                        if (
                            event.delta.type == "input_json_delta"
                            and stream_fields
                            and stream_cb
                            and current_tool_id
                            and current_tool_name
                        ):
                            json_buf += event.delta.partial_json.encode()
                            try:
                                parsed = _parse_partial_json(
                                    json_buf, partial_mode="trailing-strings"
                                )
                            except ValueError:
                                continue  # buffer isn't parseable yet; wait for more bytes

                            # Fire the tool header once, when the parsed snapshot
                            # has BOTH a populated ``reason`` and meaningful content
                            # in a streamable field. Gating on ``reason`` — not just
                            # the streamable field — avoids the race where
                            # ``thinking`` already has bytes but ``reason`` is still
                            # being parsed; that race produced ``tool → make_plan``
                            # with no label. Reason appears first in every tool
                            # schema (see ``REASON_FIELD_DESCRIPTION`` injection)
                            # so this gate adds at most a few bytes of delay.
                            reason_ready = isinstance(
                                parsed.get("reason"), str
                            ) and parsed["reason"].strip()
                            stream_ready = any(
                                isinstance(parsed.get(f), str)
                                and parsed.get(f, "").strip()
                                for f in stream_fields
                            )
                            if not header_fired and reason_ready and stream_ready:
                                on_tool_start(current_tool_id, current_tool_name, parsed)
                                header_fired = True

                            # Emit each field's rendered delta. ``_render_field``
                            # handles the section header and leading-whitespace
                            # stripping; we just track how many rendered chars
                            # have been forwarded and emit the new suffix.
                            for field in stream_fields:
                                value = parsed.get(field, "")
                                if not isinstance(value, str):
                                    continue
                                rendered = _render_field(field, value)
                                if len(rendered) > streamed[field]:
                                    stream_cb(rendered[streamed[field]:])
                                    streamed[field] = len(rendered)

                response = await stream.get_final_message()

        except APIError as e:
            logger.error("API error: %s", e)
            raise

        elapsed = time.monotonic() - start

        # Parse the response into our clean data types
        content: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        cache_creation = response.usage.cache_creation_input_tokens or 0
        cache_read = response.usage.cache_read_input_tokens or 0

        result = AgentResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

        logger.info(
            "API response: stop_reason=%s, tokens=%d/%d, "
            "cache_create=%d, cache_read=%d, tool_calls=%d, elapsed=%.1fs",
            result.stop_reason,
            result.input_tokens,
            result.output_tokens,
            cache_creation,
            cache_read,
            len(result.tool_calls),
            elapsed,
        )
        logger.debug(
            "Response content:\n%s", json.dumps(content, indent=2, default=str)
        )

        return result
