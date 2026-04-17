"""OpenAILLMClient — ``LLMClient`` adapter for any OpenAI-compatible endpoint.

Speaks the ``/v1/chat/completions`` protocol via the ``openai`` Python SDK.
"openai" here names the *wire protocol*, not the company: this adapter
works against Ollama, LM Studio, vLLM, Groq, Together, DeepSeek, and any
other server that implements the same API, selected via ``OPENAI_BASE_URL``.

Phase 1: batch mode — the full stream is consumed before any user-visible
callback fires. Live character-by-character streaming of ``make_plan`` /
``send_response`` fields is deferred to Phase 2 (see the plan at
``docs/openai-compatible-provider.md``).
"""

import json
import logging
import time
from typing import Any, Callable

from openai import APIError, AsyncOpenAI

from thesaurus.core.ports import AgentResponse, ToolCall
from thesaurus.tools.base import ToolName

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Streamable-tool rendering helpers
#
# Private duplicates of the three pieces in ``anthropic_llm.py`` (which are
# module-private and cannot be imported). Collapsing into a shared helper
# waits for a third provider per the "third occurrence" rule.
# ───────────────────────────────────────────────────────────────────────────

_STREAM_FIELDS_BY_TOOL: dict[str, list[str]] = {
    ToolName.MAKE_PLAN: ["thinking", "roadmap"],
    ToolName.SEND_RESPONSE: ["response"],
}

_FIELD_PREFIX: dict[str, str] = {
    "thinking": "## Thinking\n\n",
    "roadmap": "\n\n## Roadmap\n\n",
}


def _render_field(field: str, value: str) -> str:
    """Render a streamable field with its section-header prefix, matching
    the behavior of ``adapters.anthropic_llm._render_field`` so TUI output
    is identical regardless of provider.
    """
    stripped = value.lstrip()
    if not stripped:
        return ""
    return _FIELD_PREFIX.get(field, "") + stripped


# ───────────────────────────────────────────────────────────────────────────
# Anthropic → OpenAI translation
# ───────────────────────────────────────────────────────────────────────────


def _translate_system(system: list[dict[str, Any]]) -> dict[str, Any]:
    """Join Anthropic system text blocks into a single OpenAI system message.

    Drops ``cache_control`` (Anthropic-only extension; OpenAI-compatible
    endpoints reject unknown fields).
    """
    text = "".join(
        block.get("text", "")
        for block in system
        if block.get("type") == "text"
    )
    return {"role": "system", "content": text}


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI ``function`` form.

    Drops ``cache_control`` — ``tools_to_api_format`` in ``tools/base.py``
    attaches it to the last tool for Anthropic prompt caching, which
    OpenAI-compatible endpoints don't understand.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-shape messages to OpenAI shape.

    Two non-obvious cases:

    - **Assistant multi-block**: one Anthropic assistant message can mix
      ``text`` and ``tool_use`` blocks. Flatten into a single OpenAI
      assistant message with ``content`` (joined text) and ``tool_calls``.

    - **User multi-result**: one Anthropic user message can carry N
      ``tool_result`` blocks. OpenAI requires one ``{role: "tool"}``
      message per result, so split.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            # List of blocks — either plain text or tool_result blocks.
            plain_text: list[str] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # Anthropic nests text blocks inside tool_result.content
                        result_content = "".join(
                            b.get("text", "")
                            for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    out.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(result_content),
                    })
                elif btype == "text":
                    plain_text.append(block.get("text", ""))
            if plain_text:
                out.append({"role": "user", "content": "".join(plain_text)})

        elif role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            assistant_msg["content"] = "".join(text_parts) if text_parts else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)

    return out


# ───────────────────────────────────────────────────────────────────────────
# stop_reason mapping
# ───────────────────────────────────────────────────────────────────────────

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "stop_sequence",
}


# ───────────────────────────────────────────────────────────────────────────
# Adapter
# ───────────────────────────────────────────────────────────────────────────


class OpenAILLMClient:
    """``LLMClient`` adapter wrapping the OpenAI Python SDK.

    Works with any server implementing the OpenAI ``/v1/chat/completions``
    protocol: Ollama, LM Studio, vLLM, OpenAI, Groq, Together, DeepSeek,
    and others. Selected via ``OPENAI_BASE_URL``.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
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
        """Call the OpenAI-compatible endpoint with streaming and return a
        parsed response. Phase 1: batch mode — callbacks fire after the
        full stream is consumed.
        """
        openai_messages = [_translate_system(system), *_translate_messages(messages)]
        openai_tools = _translate_tools(tools)

        logger.info(
            "API request: model=%s, messages=%d, tools=%s",
            self._model,
            len(openai_messages),
            [t["function"]["name"] for t in openai_tools],
        )

        start = time.monotonic()

        # Accumulate state across chunks. Tool calls are indexed by their
        # ``delta.tool_calls[].index`` to handle parallel calls.
        tool_buf: dict[int, dict[str, str]] = {}
        text_parts: list[str] = []
        finish_reason: str | None = None
        usage = None

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice="required",
                max_tokens=16_384,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    text_parts.append(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        buf = tool_buf.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""},
                        )
                        if tc.id:
                            buf["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments"] += tc.function.arguments
        except APIError as e:
            logger.error("API error: %s", e)
            raise

        # Parse each accumulated tool call's arguments JSON. Fail fast on
        # malformed output — retrying would just hide a model-capability
        # problem.
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_buf.keys()):
            entry = tool_buf[idx]
            raw_args = entry["arguments"]
            try:
                parsed_input = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Model emitted invalid tool-call JSON for "
                    f"{entry['name']!r}: {e}. Try a larger/stronger model."
                ) from e
            tool_calls.append(ToolCall(
                id=entry["id"],
                name=entry["name"],
                input=parsed_input,
            ))

        # Fire callbacks for streamable tools only. Non-streamable tools
        # are rendered by the processor from ``AgentResponse.tool_calls``,
        # matching the Anthropic adapter's split of responsibility.
        for call in tool_calls:
            fields = _STREAM_FIELDS_BY_TOOL.get(call.name)
            if not fields:
                continue
            on_tool_start(call.id, call.name, call.input)
            cb = on_thinking if call.name == ToolName.MAKE_PLAN else on_text
            for field in fields:
                value = call.input.get(field, "")
                if not isinstance(value, str):
                    continue
                rendered = _render_field(field, value)
                if rendered:
                    cb(rendered)

        # Build the Anthropic-shape content list for history replay.
        text_content = "".join(text_parts)
        content: list[dict[str, Any]] = []
        if text_content:
            content.append({"type": "text", "text": text_content})
        for call in tool_calls:
            content.append({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.input,
            })

        stop_reason = _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")

        result = AgentResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "API response: stop_reason=%s, tokens=%d/%d, "
            "tool_calls=%d, elapsed=%.1fs",
            result.stop_reason,
            result.input_tokens,
            result.output_tokens,
            len(result.tool_calls),
            elapsed,
        )
        logger.debug(
            "Response content:\n%s", json.dumps(content, indent=2, default=str),
        )

        return result
