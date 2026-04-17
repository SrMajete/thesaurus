"""Tests for the OpenAI-compatible LLM adapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from thesaurus.adapters.openai_llm import (
    OpenAILLMClient,
    _render_field,
    _translate_messages,
    _translate_system,
    _translate_tools,
)
from thesaurus.core.ports import AgentResponse


# ───────────────────────────────────────────────────────────────────────────
# _render_field
# ───────────────────────────────────────────────────────────────────────────


class TestRenderField:
    def test_empty_value_returns_empty(self) -> None:
        assert _render_field("thinking", "") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert _render_field("thinking", "   \n\n") == ""

    def test_thinking_has_header(self) -> None:
        assert _render_field("thinking", "hello") == "## Thinking\n\nhello"

    def test_roadmap_has_header(self) -> None:
        r = _render_field("roadmap", "1. step")
        assert r.startswith("\n\n## Roadmap\n\n")

    def test_response_has_no_prefix(self) -> None:
        assert _render_field("response", "hi") == "hi"

    def test_leading_whitespace_stripped(self) -> None:
        assert _render_field("response", "\n\n  hi") == "hi"


# ───────────────────────────────────────────────────────────────────────────
# _translate_system / _translate_tools
# ───────────────────────────────────────────────────────────────────────────


class TestTranslateSystem:
    def test_joins_text_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "You are "},
            {"type": "text", "text": "an agent."},
        ]
        assert _translate_system(blocks) == {"role": "system", "content": "You are an agent."}

    def test_drops_cache_control(self) -> None:
        blocks = [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
        ]
        out = _translate_system(blocks)
        assert out == {"role": "system", "content": "hi"}

    def test_ignores_non_text_blocks(self) -> None:
        blocks = [{"type": "image", "source": {}}, {"type": "text", "text": "ok"}]
        assert _translate_system(blocks)["content"] == "ok"

    def test_empty_list(self) -> None:
        assert _translate_system([]) == {"role": "system", "content": ""}


class TestTranslateTools:
    def test_converts_to_function_shape(self) -> None:
        tools = [{
            "name": "read_file",
            "description": "reads a file",
            "input_schema": {"type": "object", "properties": {}},
        }]
        out = _translate_tools(tools)
        assert out == [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "reads a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def test_drops_cache_control(self) -> None:
        """cache_control is attached to the last tool by tools_to_api_format
        in tools/base.py; it must not appear in the OpenAI request body."""
        tools = [{
            "name": "t",
            "description": "d",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        }]
        out = _translate_tools(tools)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[0]["function"]

    def test_empty_list(self) -> None:
        assert _translate_tools([]) == []


# ───────────────────────────────────────────────────────────────────────────
# _translate_messages
# ───────────────────────────────────────────────────────────────────────────


class TestTranslateMessages:
    def test_user_string_passthrough(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        assert _translate_messages(msgs) == [{"role": "user", "content": "hello"}]

    def test_assistant_string_passthrough(self) -> None:
        msgs = [{"role": "assistant", "content": "hi"}]
        assert _translate_messages(msgs) == [{"role": "assistant", "content": "hi"}]

    def test_single_tool_result_becomes_tool_message(self) -> None:
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]}]
        out = _translate_messages(msgs)
        assert out == [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]

    def test_multi_tool_result_split_into_n_messages(self) -> None:
        """Anthropic allows one user message with N tool_result blocks;
        OpenAI requires N separate {role: 'tool'} messages."""
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
            {"type": "tool_result", "tool_use_id": "t3", "content": "c"},
        ]}]
        out = _translate_messages(msgs)
        assert len(out) == 3
        assert [m["role"] for m in out] == ["tool", "tool", "tool"]
        assert [m["tool_call_id"] for m in out] == ["t1", "t2", "t3"]
        assert [m["content"] for m in out] == ["a", "b", "c"]

    def test_tool_result_with_nested_text_content(self) -> None:
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ]},
        ]}]
        out = _translate_messages(msgs)
        assert out[0]["content"] == "line1line2"

    def test_user_text_blocks_collected(self) -> None:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]}]
        out = _translate_messages(msgs)
        assert out == [{"role": "user", "content": "hello world"}]

    def test_assistant_multi_block_flattens_text_and_tool_calls(self) -> None:
        """One Anthropic assistant message with mixed text + tool_use blocks
        flattens into one OpenAI message with content + tool_calls array."""
        msgs = [{"role": "assistant", "content": [
            {"type": "text", "text": "Thinking..."},
            {"type": "tool_use", "id": "t1", "name": "read_file",
             "input": {"path": "/a"}},
            {"type": "tool_use", "id": "t2", "name": "read_file",
             "input": {"path": "/b"}},
        ]}]
        out = _translate_messages(msgs)
        assert len(out) == 1
        msg = out[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Thinking..."
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["id"] == "t1"
        assert msg["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "/a"}
        assert msg["tool_calls"][1]["id"] == "t2"

    def test_assistant_with_only_tool_use_has_null_content(self) -> None:
        msgs = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}},
        ]}]
        out = _translate_messages(msgs)
        assert out[0]["content"] is None
        assert len(out[0]["tool_calls"]) == 1

    def test_empty_list(self) -> None:
        assert _translate_messages([]) == []

    def test_user_list_with_only_non_text_non_tool_result_blocks(self) -> None:
        """User content with no tool_result and no text blocks yields no output."""
        msgs = [{"role": "user", "content": [{"type": "image", "source": {}}]}]
        assert _translate_messages(msgs) == []

    def test_assistant_list_with_no_text_or_tool_use(self) -> None:
        """Assistant content with only unknown block types yields a message
        with content=None and no tool_calls."""
        msgs = [{"role": "assistant", "content": [{"type": "image", "source": {}}]}]
        out = _translate_messages(msgs)
        assert out == [{"role": "assistant", "content": None}]


# ───────────────────────────────────────────────────────────────────────────
# OpenAILLMClient.stream_response — mocked integration tests
# ───────────────────────────────────────────────────────────────────────────


def _delta_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> MagicMock:
    """Build a fake ``ChatCompletionChunk``.

    Each tool-call dict in ``tool_calls`` may carry ``index``, ``id``,
    ``function`` (with ``name`` / ``arguments``). Missing keys become
    ``None`` on the underlying mock.
    """
    chunk = MagicMock()
    chunk.usage = None
    if usage:
        chunk.usage = MagicMock(**usage)

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.delta = MagicMock()
    choice.delta.content = content
    if tool_calls is None:
        choice.delta.tool_calls = None
    else:
        tc_mocks = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.index = tc.get("index", 0)
            tc_mock.id = tc.get("id")
            if "function" in tc:
                tc_mock.function = MagicMock()
                tc_mock.function.name = tc["function"].get("name")
                tc_mock.function.arguments = tc["function"].get("arguments")
            else:
                tc_mock.function = None
            tc_mocks.append(tc_mock)
        choice.delta.tool_calls = tc_mocks
    chunk.choices = [choice]
    return chunk


def _mock_client_with_chunks(chunks: list) -> MagicMock:
    """Build a mock ``AsyncOpenAI`` whose ``chat.completions.create`` returns
    an async iterator yielding the given chunks."""
    async def aiter():
        for c in chunks:
            yield c

    class _Stream:
        def __aiter__(self):
            return aiter()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_Stream())
    return client


def _make_adapter_with_chunks(chunks: list) -> tuple[OpenAILLMClient, MagicMock]:
    """Create an OpenAILLMClient whose SDK is mocked to return the chunks."""
    adapter = OpenAILLMClient(base_url="http://x", api_key="k", model="m")
    mock = _mock_client_with_chunks(chunks)
    adapter._client = mock
    return adapter, mock


class TestStreamResponse:
    async def test_text_only_response(self) -> None:
        chunks = [
            _delta_chunk(content="Hello "),
            _delta_chunk(content="world"),
            _delta_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            ),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.stop_reason == "end_turn"
        assert r.content == [{"type": "text", "text": "Hello world"}]
        assert r.tool_calls == []
        assert r.input_tokens == 5
        assert r.output_tokens == 2
        assert r.cache_creation_input_tokens == 0
        assert r.cache_read_input_tokens == 0

    async def test_single_tool_call_uses_server_id(self) -> None:
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "call_xyz",
                "function": {"name": "read_file", "arguments": ""},
            }]),
            _delta_chunk(tool_calls=[{
                "index": 0,
                "function": {"arguments": '{"reason":"r","path":"/a"}'},
            }]),
            _delta_chunk(
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            ),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.stop_reason == "tool_use"
        assert len(r.tool_calls) == 1
        # Server-provided ID round-trips unchanged (no uuid4 replacement).
        assert r.tool_calls[0].id == "call_xyz"
        assert r.tool_calls[0].name == "read_file"
        assert r.tool_calls[0].input == {"reason": "r", "path": "/a"}

    async def test_multiple_tool_calls_accumulate_by_index(self) -> None:
        """Parallel tool calls arrive interleaved via delta.tool_calls[].index;
        each index's arguments must be accumulated separately."""
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "read_file", "arguments": '{"path":"/'},
            }]),
            _delta_chunk(tool_calls=[{
                "index": 1, "id": "c2",
                "function": {"name": "read_file", "arguments": '{"path":"/'},
            }]),
            _delta_chunk(tool_calls=[{
                "index": 0, "function": {"arguments": 'a"}'},
            }]),
            _delta_chunk(tool_calls=[{
                "index": 1, "function": {"arguments": 'b"}'},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert len(r.tool_calls) == 2
        assert r.tool_calls[0].input == {"path": "/a"}
        assert r.tool_calls[1].input == {"path": "/b"}

    async def test_streamable_tool_fires_callbacks(self) -> None:
        args = json.dumps({
            "reason": "answering",
            "response": "Here is the answer.",
        })
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "send_response", "arguments": args},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        texts: list[str] = []
        thinkings: list[str] = []
        tool_starts: list[tuple[str, str, dict]] = []
        await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: texts.append(t),
            on_thinking=lambda t: thinkings.append(t),
            on_tool_start=lambda *a: tool_starts.append(a),
        )
        assert tool_starts == [("c1", "send_response", {
            "reason": "answering",
            "response": "Here is the answer.",
        })]
        assert texts == ["Here is the answer."]
        assert thinkings == []

    async def test_make_plan_streams_both_fields_to_on_thinking(self) -> None:
        args = json.dumps({
            "reason": "planning",
            "thinking": "Need to plan.",
            "roadmap": "1. Do X",
        })
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "make_plan", "arguments": args},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        texts: list[str] = []
        thinkings: list[str] = []
        await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: texts.append(t),
            on_thinking=lambda t: thinkings.append(t),
            on_tool_start=lambda *a: None,
        )
        # Both thinking AND roadmap route to on_thinking (per-tool, not per-field).
        assert texts == []
        assert len(thinkings) == 2
        assert thinkings[0] == "## Thinking\n\nNeed to plan."
        assert thinkings[1].startswith("\n\n## Roadmap\n\n")
        assert "1. Do X" in thinkings[1]

    async def test_non_streamable_tool_does_not_fire_on_tool_start(self) -> None:
        args = json.dumps({"reason": "r", "path": "/a"})
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "read_file", "arguments": args},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        tool_starts: list[tuple] = []
        await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: tool_starts.append(a),
        )
        # read_file is not streamable — processor handles it post-adapter.
        assert tool_starts == []

    async def test_malformed_json_raises_fail_fast(self) -> None:
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "read_file", "arguments": "{not-json"},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        with pytest.raises(ValueError, match="invalid tool-call JSON"):
            await adapter.stream_response(
                messages=[], system=[], tools=[],
                on_text=lambda t: None,
                on_thinking=lambda t: None,
                on_tool_start=lambda *a: None,
            )

    async def test_empty_arguments_parse_to_empty_dict(self) -> None:
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "read_file", "arguments": ""},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.tool_calls[0].input == {}

    async def test_request_is_built_correctly(self) -> None:
        chunks = [_delta_chunk(content="x", finish_reason="stop")]
        adapter, mock_client = _make_adapter_with_chunks(chunks)
        await adapter.stream_response(
            messages=[{"role": "user", "content": "hello"}],
            system=[{"type": "text", "text": "sys"}],
            tools=[{
                "name": "read_file", "description": "r",
                "input_schema": {"type": "object"},
            }],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "m"
        assert call_kwargs["tool_choice"] == "required"
        assert call_kwargs["max_tokens"] == 16_384
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}
        # System message prepended
        assert call_kwargs["messages"][0] == {"role": "system", "content": "sys"}
        # Tools translated to function shape
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "read_file"

    async def test_missing_usage_produces_zero_counts(self) -> None:
        chunks = [_delta_chunk(content="x", finish_reason="stop")]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    async def test_length_finish_reason_maps_to_max_tokens(self) -> None:
        chunks = [_delta_chunk(content="x", finish_reason="length")]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.stop_reason == "max_tokens"

    async def test_unknown_finish_reason_maps_to_end_turn(self) -> None:
        chunks = [_delta_chunk(content="x", finish_reason="weird_value")]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.stop_reason == "end_turn"

    async def test_streamable_tool_with_non_string_field_is_skipped(self) -> None:
        """If the model returns a non-string value for a streamable field,
        don't crash — just skip emitting that field."""
        args = json.dumps({"reason": "r", "response": None})
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "send_response", "arguments": args},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        texts: list[str] = []
        await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: texts.append(t),
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert texts == []

    async def test_streamable_tool_with_whitespace_field_emits_nothing(self) -> None:
        """Whitespace-only field value renders to empty string — no callback."""
        args = json.dumps({"reason": "r", "response": "   \n\n"})
        chunks = [
            _delta_chunk(tool_calls=[{
                "index": 0, "id": "c1",
                "function": {"name": "send_response", "arguments": args},
            }]),
            _delta_chunk(finish_reason="tool_calls"),
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        texts: list[str] = []
        await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: texts.append(t),
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert texts == []

    async def test_chunk_with_no_choices_is_skipped(self) -> None:
        """Some providers emit a final usage-only chunk with empty choices."""
        no_choice_chunk = MagicMock()
        no_choice_chunk.choices = []
        no_choice_chunk.usage = MagicMock(prompt_tokens=3, completion_tokens=1)
        chunks = [
            _delta_chunk(content="x", finish_reason="stop"),
            no_choice_chunk,
        ]
        adapter, _ = _make_adapter_with_chunks(chunks)
        r = await adapter.stream_response(
            messages=[], system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert r.input_tokens == 3
        assert r.output_tokens == 1
        assert r.content == [{"type": "text", "text": "x"}]

    async def test_api_error_logged_and_reraised(self, caplog) -> None:
        """APIError from the SDK is logged at ERROR level and re-raised —
        observability parity with the Anthropic adapter.
        """
        import httpx
        from openai import APIError

        adapter = OpenAILLMClient(base_url="http://x", api_key="k", model="m")
        adapter._client = MagicMock()
        fake_request = httpx.Request("POST", "https://x/v1/chat/completions")
        adapter._client.chat.completions.create = AsyncMock(
            side_effect=APIError("boom", request=fake_request, body=None),
        )
        with caplog.at_level("ERROR"), pytest.raises(APIError):
            await adapter.stream_response(
                messages=[], system=[], tools=[],
                on_text=lambda t: None,
                on_thinking=lambda t: None,
                on_tool_start=lambda *a: None,
            )
        assert "API error" in caplog.text
