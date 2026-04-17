"""Tests for api_client: data types, pure helpers, and mocked streaming."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from thesaurus.core.ports import AgentResponse, ToolCall
from thesaurus.adapters.anthropic_llm import (
    AnthropicLLMClient,
    _render_field,
    _with_message_cache_breakpoint,
)


class TestAgentResponse:
    def test_context_tokens_sum(self) -> None:
        r = AgentResponse(
            content=[], tool_calls=[], stop_reason="end_turn",
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )
        assert r.context_tokens == 200  # 100+20+30+50

    def test_frozen(self) -> None:
        r = AgentResponse(
            content=[], tool_calls=[], stop_reason="end_turn",
            input_tokens=0, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        with pytest.raises(Exception):  # dataclass FrozenInstanceError
            r.input_tokens = 999  # type: ignore[misc]


class TestToolCall:
    def test_frozen_dataclass(self) -> None:
        tc = ToolCall(id="x", name="read_file", input={"path": "/a"})
        assert tc.id == "x"
        with pytest.raises(Exception):
            tc.id = "y"  # type: ignore[misc]


class TestRenderField:
    def test_empty_value_returns_empty(self) -> None:
        assert _render_field("thinking", "") == ""
        assert _render_field("thinking", "   \n ") == ""

    def test_thinking_has_header(self) -> None:
        r = _render_field("thinking", "hello")
        assert r == "## Thinking\n\nhello"

    def test_roadmap_has_header(self) -> None:
        r = _render_field("roadmap", "1. step")
        assert r.startswith("\n\n## Roadmap\n\n")

    def test_no_prefix_field(self) -> None:
        assert _render_field("response", "hi") == "hi"

    def test_leading_whitespace_stripped(self) -> None:
        r = _render_field("response", "\n\n hi")
        assert r == "hi"


class TestWithMessageCacheBreakpoint:
    def test_no_user_messages_unchanged(self) -> None:
        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        out = _with_message_cache_breakpoint(msgs)
        assert out == msgs

    def test_string_content_user_unchanged(self) -> None:
        """User messages with string content can't carry cache_control blocks."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        out = _with_message_cache_breakpoint(msgs)
        assert out[0]["content"] == "hello"

    def test_adds_breakpoint_to_last_user_list_message(self) -> None:
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "2", "content": "ok"},
            ]},
        ]
        out = _with_message_cache_breakpoint(msgs)
        last_block = out[0]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}
        # First block untouched
        assert "cache_control" not in out[0]["content"][0]

    def test_does_not_mutate_original(self) -> None:
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "ok"},
            ]},
        ]
        _with_message_cache_breakpoint(msgs)
        assert "cache_control" not in msgs[0]["content"][0]

    def test_picks_last_user_message(self) -> None:
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "old"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "new"},
            ]},
        ]
        out = _with_message_cache_breakpoint(msgs)
        # Older user message untouched
        assert "cache_control" not in out[0]["content"][0]
        # Newer user message tagged
        assert out[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestStreamResponse:
    """Integration-style tests with the Anthropic stream mocked."""

    async def test_parses_text_and_tool_calls(self) -> None:
        final = self._final_message(
            content=[
                MagicMock(type="text", text="hi"),
                self._tool_use_block(
                    id="t1", name="read_file", input={"file_path": "/a"},
                ),
            ],
            usage=MagicMock(
                input_tokens=10, output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            stop_reason="tool_use",
        )
        client = self._mock_client_returning([], final)

        result = await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[{"type": "text", "text": "sys"}],
            tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert result.stop_reason == "tool_use"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.content[0] == {"type": "text", "text": "hi"}

    async def test_api_error_propagates(self) -> None:
        from anthropic import APIError

        client = MagicMock()

        class Ctx:
            async def __aenter__(self):
                raise APIError(
                    "boom", request=MagicMock(), body=None,
                )

            async def __aexit__(self, *args):
                return False

        client.messages.stream = MagicMock(return_value=Ctx())
        with pytest.raises(APIError):
            await AnthropicLLMClient(client=client, model="m").stream_response( messages=[],
                system=[], tools=[],
                on_text=lambda t: None,
                on_thinking=lambda t: None,
                on_tool_start=lambda *a: None,
            )

    async def test_null_cache_tokens_treated_as_zero(self) -> None:
        final = self._final_message(
            content=[MagicMock(type="text", text="hi")],
            usage=MagicMock(
                input_tokens=10, output_tokens=5,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
            stop_reason="end_turn",
        )
        client = self._mock_client_returning([], final)
        result = await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    async def test_streams_make_plan_thinking(self) -> None:
        """Verify on_thinking receives the streamed thinking field."""
        make_plan_id = "plan1"
        events = [
            self._event("content_block_start", content_block=self._tool_use_block(
                id=make_plan_id, name="make_plan", input={},
            )),
            # Partial JSON that builds up a reason + thinking
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json='{"reason": "planning", "thinking": "step',
            )),
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json=' 1"}',
            )),
            self._event("content_block_stop"),
        ]
        final = self._final_message(
            content=[self._tool_use_block(
                id=make_plan_id, name="make_plan",
                input={"reason": "planning", "thinking": "step 1"},
            )],
            usage=MagicMock(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            stop_reason="tool_use",
        )
        client = self._mock_client_returning(events, final)

        thinking_chunks: list[str] = []
        tool_starts: list[tuple[str, str, dict]] = []
        await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=thinking_chunks.append,
            on_tool_start=lambda tid, name, params: tool_starts.append(
                (tid, name, params),
            ),
        )
        # Thinking should have been streamed
        assert "".join(thinking_chunks).endswith("step 1")
        # Tool header should fire once
        assert len(tool_starts) == 1
        assert tool_starts[0][1] == "make_plan"

    async def test_non_streamable_tool_use_no_streaming(self) -> None:
        """A non-streamable tool's tool_use block doesn't trigger streaming."""
        events = [
            self._event("content_block_start", content_block=self._tool_use_block(
                id="r1", name="read_file", input={},
            )),
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json='{"reason": "x", "file_path": "/a"}',
            )),
            self._event("content_block_stop"),
        ]
        final = self._final_message(
            content=[self._tool_use_block(
                id="r1", name="read_file",
                input={"reason": "x", "file_path": "/a"},
            )],
            usage=MagicMock(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            stop_reason="tool_use",
        )
        client = self._mock_client_returning(events, final)

        chunks: list[str] = []
        await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=chunks.append,
            on_thinking=chunks.append,
            on_tool_start=lambda *a: None,
        )
        assert chunks == []  # nothing was streamed

    async def test_partial_json_unparseable_continues(self) -> None:
        """Invalid JSON fragments are silently skipped until more bytes arrive."""
        make_plan_id = "plan1"
        events = [
            self._event("content_block_start", content_block=self._tool_use_block(
                id=make_plan_id, name="make_plan", input={},
            )),
            # Invalid partial — no key/value to parse even partially
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json="not json at all !!!",
            )),
            # Then a valid chunk completes the payload
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json='{"reason": "r", "thinking": "t"}',
            )),
            self._event("content_block_stop"),
        ]
        final = self._final_message(
            content=[self._tool_use_block(
                id=make_plan_id, name="make_plan",
                input={"reason": "r", "thinking": "t"},
            )],
            usage=MagicMock(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            stop_reason="tool_use",
        )
        client = self._mock_client_returning(events, final)
        # Should not raise despite the invalid JSON fragment
        result = await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert result.stop_reason == "tool_use"

    async def test_non_string_field_value_skipped(self) -> None:
        """If a streamable field parses as a non-string, skip it."""
        make_plan_id = "plan1"
        # Models shouldn't produce this, but the defensive branch exists.
        # Construct partial JSON where thinking parses as a number.
        events = [
            self._event("content_block_start", content_block=self._tool_use_block(
                id=make_plan_id, name="make_plan", input={},
            )),
            self._event("content_block_delta", delta=MagicMock(
                type="input_json_delta",
                partial_json='{"reason": "r", "thinking": 42, "roadmap": "r"}',
            )),
            self._event("content_block_stop"),
        ]
        final = self._final_message(
            content=[self._tool_use_block(
                id=make_plan_id, name="make_plan",
                input={"reason": "r", "thinking": 42, "roadmap": "r"},
            )],
            usage=MagicMock(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            stop_reason="tool_use",
        )
        client = self._mock_client_returning(events, final)
        chunks: list[str] = []
        await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=chunks.append,
            on_tool_start=lambda *a: None,
        )
        # Roadmap should still have been streamed, but thinking (non-str) skipped
        joined = "".join(chunks)
        assert "42" not in joined

    async def test_text_block_in_response(self) -> None:
        """A top-level text block is captured in content."""
        final = self._final_message(
            content=[MagicMock(type="text", text="hello world")],
            usage=MagicMock(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            stop_reason="end_turn",
        )
        client = self._mock_client_returning([], final)
        result = await AnthropicLLMClient(client=client, model="m").stream_response(
            messages=[{"role": "user", "content": "hi"}],
            system=[], tools=[],
            on_text=lambda t: None,
            on_thinking=lambda t: None,
            on_tool_start=lambda *a: None,
        )
        assert result.content == [{"type": "text", "text": "hello world"}]

    @staticmethod
    def _tool_use_block(*, id: str, name: str, input: dict) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.id = id
        block.name = name
        block.input = input
        return block

    @staticmethod
    def _event(type_: str, **kwargs) -> MagicMock:  # type: ignore[no-untyped-def]
        ev = MagicMock()
        ev.type = type_
        for k, v in kwargs.items():
            setattr(ev, k, v)
        return ev

    @staticmethod
    def _final_message(
        *, content: list, usage: MagicMock, stop_reason: str,
    ) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        msg.usage = usage
        msg.stop_reason = stop_reason
        return msg

    @staticmethod
    def _mock_client_returning(
        events: list, final_message: MagicMock,
    ) -> MagicMock:
        client = MagicMock()

        class Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def __aiter__(self):
                async def gen():
                    for e in events:
                        yield e
                return gen()

            async def get_final_message(self):
                return final_message

        client.messages.stream = MagicMock(return_value=Stream())
        return client
