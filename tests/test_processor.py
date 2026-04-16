"""Tests for the processor loop and its helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python_agent.api_client import AgentResponse, ToolCall
from python_agent.processor import (
    _build_send_response_results,
    _collect_decisions,
    _execute_decisions,
    _execute_tool_calls,
    _format_plan,
    _handle_plan,
    _partition,
    _strip_reason,
    run_loop,
)


class _FakeTool:
    def __init__(
        self, name: str,
        needs_permission: bool = False,
        is_parallelizable: bool = True,
        execute_result: str = "ok",
        raises: bool = False,
    ) -> None:
        self.name = name
        self.needs_permission = needs_permission
        self.is_parallelizable = is_parallelizable
        self.is_intercepted = False
        self._result = execute_result
        self._raises = raises
        self.execute_calls: list[dict[str, Any]] = []

    description = "fake"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **params: Any) -> str:
        self.execute_calls.append(params)
        if self._raises:
            raise RuntimeError("boom")
        return self._result


def _mock_agent(
    tools: list[_FakeTool] | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_context_tokens: int = 200_000,
    prune_context_threshold: float = 0.8,
) -> MagicMock:
    agent = MagicMock()
    agent.tools = tools or []
    agent.messages = messages if messages is not None else []
    agent.plan = None
    agent.total_input_tokens = 0
    agent.total_cached_input_tokens = 0
    agent.total_output_tokens = 0
    agent.last_context_tokens = 0
    agent.max_context_tokens = max_context_tokens
    agent.prune_context_threshold = prune_context_threshold
    agent.callbacks = MagicMock()
    agent.callbacks.on_thinking = MagicMock()
    agent.callbacks.on_text = MagicMock()
    agent.callbacks.on_tool_start = MagicMock()
    agent.callbacks.on_tool_result = MagicMock()
    agent.callbacks.ask_permission = AsyncMock(return_value=True)
    return agent


class TestFormatPlan:
    def test_thinking_only(self) -> None:
        assert _format_plan("hello", "") == "## Thinking\n\nhello"

    def test_roadmap_only(self) -> None:
        assert _format_plan("", "1. step") == "## Roadmap\n\n1. step"

    def test_both(self) -> None:
        r = _format_plan("t", "r")
        assert "## Thinking\n\nt" in r
        assert "## Roadmap\n\nr" in r
        assert "\n\n" in r  # separator

    def test_empty(self) -> None:
        assert _format_plan("", "") == ""

    def test_strips_whitespace(self) -> None:
        r = _format_plan("  t  ", "  r  ")
        assert "## Thinking\n\nt" in r
        assert "## Roadmap\n\nr" in r


class TestHandlePlan:
    def test_empty_list_noop(self) -> None:
        agent = _mock_agent()
        _handle_plan(agent, [])
        assert agent.plan is None

    def test_uses_first_plan_call(self) -> None:
        agent = _mock_agent()
        calls = [
            ToolCall(id="p1", name="make_plan",
                     input={"thinking": "first", "roadmap": "a"}),
            ToolCall(id="p2", name="make_plan",
                     input={"thinking": "second", "roadmap": "b"}),
        ]
        _handle_plan(agent, calls)
        assert "first" in agent.plan
        assert "second" not in agent.plan

    def test_empty_plan_not_stored(self) -> None:
        agent = _mock_agent()
        _handle_plan(agent, [
            ToolCall(id="p1", name="make_plan", input={}),
        ])
        # _format_plan returns "" which is falsy; logger line skipped but
        # agent.plan is still set to ""
        assert agent.plan == ""


class TestBuildSendResponseResults:
    def test_respond_only(self) -> None:
        agent = _mock_agent()
        calls = [
            ToolCall(id="r1", name="send_response",
                     input={"response": "final answer"}),
        ]
        results = _build_send_response_results(
            agent, calls, plan_ids=set(), respond_ids={"r1"},
        )
        assert len(results) == 1
        assert results[0]["content"] == "final answer"
        assert results[0]["tool_use_id"] == "r1"

    def test_plan_and_respond(self) -> None:
        agent = _mock_agent()
        calls = [
            ToolCall(id="p1", name="make_plan", input={}),
            ToolCall(id="r1", name="send_response",
                     input={"response": "done"}),
        ]
        results = _build_send_response_results(
            agent, calls, plan_ids={"p1"}, respond_ids={"r1"},
        )
        assert results[0]["content"] == "Plan acknowledged."
        assert results[1]["content"] == "done"

    def test_extra_tools_skipped_with_callbacks(self) -> None:
        agent = _mock_agent()
        calls = [
            ToolCall(id="x1", name="read_file", input={"file_path": "/a"}),
            ToolCall(id="r1", name="send_response", input={"response": "hi"}),
        ]
        results = _build_send_response_results(
            agent, calls, plan_ids=set(), respond_ids={"r1"},
        )
        assert len(results) == 2
        assert "Skipped" in results[0]["content"]
        # Both callbacks fired for the skipped tool (with is_error=True to the UI)
        assert agent.callbacks.on_tool_start.call_count == 1
        assert agent.callbacks.on_tool_result.call_count == 1
        # Callback received is_error=True so the UI can show the skip as an error
        assert agent.callbacks.on_tool_result.call_args.args[-1] is True


class TestStripReason:
    def test_removes_reason(self) -> None:
        assert _strip_reason({"reason": "x", "y": 1}) == {"y": 1}

    def test_no_reason_unchanged(self) -> None:
        assert _strip_reason({"y": 1}) == {"y": 1}

    def test_empty(self) -> None:
        assert _strip_reason({}) == {}


class TestPartition:
    def test_empty(self) -> None:
        agent = _mock_agent()
        assert _partition(agent, []) == []

    def test_grouping_by_flags(self) -> None:
        read = _FakeTool("read_file", needs_permission=False, is_parallelizable=True)
        python = _FakeTool("run_python", needs_permission=True, is_parallelizable=True)
        write = _FakeTool("write_file", needs_permission=True, is_parallelizable=False)
        agent = _mock_agent(tools=[read, python, write])

        calls = [
            ToolCall(id="1", name="read_file", input={}),
            ToolCall(id="2", name="read_file", input={}),
            ToolCall(id="3", name="run_python", input={}),
            ToolCall(id="4", name="write_file", input={}),
            ToolCall(id="5", name="read_file", input={}),
        ]
        batches = _partition(agent, calls)
        # Expect: [read,read], [python], [write], [read]
        assert len(batches) == 4
        assert len(batches[0][1]) == 2
        assert batches[1][0] == (True, True)
        assert batches[2][0] == (True, False)

    def test_unknown_tool_defaults_to_approve_serial(self) -> None:
        agent = _mock_agent()  # no tools
        calls = [ToolCall(id="1", name="unknown", input={})]
        batches = _partition(agent, calls)
        assert batches[0][0] == (True, False)


class TestCollectDecisions:
    async def test_auto_path(self) -> None:
        tool = _FakeTool("read_file")
        agent = _mock_agent(tools=[tool])
        calls = [ToolCall(id="1", name="read_file", input={})]
        decisions = await _collect_decisions(agent, calls, needs_permission=False)
        assert len(decisions) == 1
        assert decisions[0][1] is tool
        assert decisions[0][2] is True
        agent.callbacks.ask_permission.assert_not_awaited()

    async def test_approve_path(self) -> None:
        tool = _FakeTool("run_bash", needs_permission=True)
        agent = _mock_agent(tools=[tool])
        calls = [ToolCall(id="1", name="run_bash", input={})]
        decisions = await _collect_decisions(agent, calls, needs_permission=True)
        agent.callbacks.ask_permission.assert_awaited_once()
        assert decisions[0][2] is True

    async def test_permission_denied(self) -> None:
        tool = _FakeTool("run_bash", needs_permission=True)
        agent = _mock_agent(tools=[tool])
        agent.callbacks.ask_permission = AsyncMock(return_value=False)
        calls = [ToolCall(id="1", name="run_bash", input={})]
        decisions = await _collect_decisions(agent, calls, needs_permission=True)
        assert decisions[0][2] is False

    async def test_unknown_tool_gets_none(self) -> None:
        agent = _mock_agent()
        calls = [ToolCall(id="1", name="missing", input={})]
        decisions = await _collect_decisions(agent, calls, needs_permission=True)
        assert decisions[0][1] is None
        assert decisions[0][2] is False


class TestExecuteDecisions:
    async def test_successful_auto_execution(self) -> None:
        tool = _FakeTool("read_file", execute_result="CONTENT")
        agent = _mock_agent(tools=[tool])
        decisions = [(
            ToolCall(id="1", name="read_file", input={"reason": "x", "p": 1}),
            tool, True,
        )]
        results = await _execute_decisions(
            agent, decisions, needs_permission=False, is_parallelizable=True,
        )
        assert results[0]["content"] == "CONTENT"
        # on_tool_start fires on auto path
        agent.callbacks.on_tool_start.assert_called_once()
        agent.callbacks.on_tool_result.assert_called_once()
        # reason stripped before execute
        assert tool.execute_calls == [{"p": 1}]

    async def test_approve_path_skips_on_tool_start(self) -> None:
        tool = _FakeTool("run_bash", needs_permission=True)
        agent = _mock_agent(tools=[tool])
        decisions = [(
            ToolCall(id="1", name="run_bash", input={}),
            tool, True,
        )]
        await _execute_decisions(
            agent, decisions, needs_permission=True, is_parallelizable=True,
        )
        # on_tool_start is NOT called on approve path
        agent.callbacks.on_tool_start.assert_not_called()

    async def test_denied_decision(self) -> None:
        tool = _FakeTool("run_bash")
        agent = _mock_agent(tools=[tool])
        decisions = [(
            ToolCall(id="1", name="run_bash", input={}),
            tool, False,
        )]
        results = await _execute_decisions(
            agent, decisions, needs_permission=True, is_parallelizable=False,
        )
        assert "denied" in results[0]["content"].lower()
        assert results[0].get("is_error") is True

    async def test_unknown_tool_decision(self) -> None:
        agent = _mock_agent()
        decisions = [(
            ToolCall(id="1", name="missing", input={}),
            None, False,
        )]
        results = await _execute_decisions(
            agent, decisions, needs_permission=True, is_parallelizable=False,
        )
        assert "unknown" in results[0]["content"].lower()
        assert results[0].get("is_error") is True

    async def test_tool_raises_becomes_error_result(self) -> None:
        tool = _FakeTool("read_file", raises=True)
        agent = _mock_agent(tools=[tool])
        decisions = [(
            ToolCall(id="1", name="read_file", input={}),
            tool, True,
        )]
        results = await _execute_decisions(
            agent, decisions, needs_permission=False, is_parallelizable=True,
        )
        assert "boom" in results[0]["content"]
        assert results[0].get("is_error") is True

    async def test_parallel_execution(self) -> None:
        tool = _FakeTool("read_file")
        agent = _mock_agent(tools=[tool])
        decisions = [
            (ToolCall(id=str(i), name="read_file", input={}), tool, True)
            for i in range(3)
        ]
        results = await _execute_decisions(
            agent, decisions, needs_permission=False, is_parallelizable=True,
        )
        assert len(results) == 3

    async def test_serial_execution(self) -> None:
        tool = _FakeTool("write_file", is_parallelizable=False)
        agent = _mock_agent(tools=[tool])
        decisions = [
            (ToolCall(id="1", name="write_file", input={}), tool, True),
            (ToolCall(id="2", name="write_file", input={}), tool, True),
        ]
        results = await _execute_decisions(
            agent, decisions, needs_permission=True, is_parallelizable=False,
        )
        assert len(results) == 2


class TestExecuteToolCalls:
    async def test_empty(self) -> None:
        agent = _mock_agent()
        results = await _execute_tool_calls(agent, [])
        assert results == []

    async def test_mixed_tools(self) -> None:
        read = _FakeTool("read_file", execute_result="READ")
        write = _FakeTool("write_file", needs_permission=True, is_parallelizable=False,
                          execute_result="WROTE")
        agent = _mock_agent(tools=[read, write])
        calls = [
            ToolCall(id="1", name="read_file", input={}),
            ToolCall(id="2", name="write_file", input={}),
        ]
        results = await _execute_tool_calls(agent, calls)
        assert results[0]["content"] == "READ"
        assert results[1]["content"] == "WROTE"


class TestRunLoop:
    async def test_send_response_exits_loop(self, monkeypatch) -> None:
        agent = _mock_agent(messages=[{"role": "user", "content": "hi"}])
        agent.max_turns = 5
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[{"type": "tool_use", "id": "r1",
                      "name": "send_response", "input": {"response": "done"}}],
            tool_calls=[
                ToolCall(id="r1", name="send_response",
                         input={"response": "done"}),
            ],
            stop_reason="tool_use",
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        calls_made = [0]

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            calls_made[0] += 1
            return response

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)

        await run_loop(agent)
        assert calls_made[0] == 1  # only one turn before send_response exits
        # Original user message + assistant content + tool_results
        assert len(agent.messages) == 3
        assert agent.last_context_tokens == 15

    async def test_no_tool_calls_ends(self, monkeypatch) -> None:
        """If the model returns no tool calls (fallback), the loop exits."""
        agent = _mock_agent(messages=[{"role": "user", "content": "hi"}])
        agent.max_turns = 5
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[{"type": "text", "text": "hi"}],
            tool_calls=[],
            stop_reason="end_turn",
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        await run_loop(agent)
        # Original user message + assistant content
        assert len(agent.messages) == 2

    async def test_tool_execution_then_send_response(self, monkeypatch) -> None:
        tool = _FakeTool("read_file", execute_result="FILE_CONTENT")
        agent = _mock_agent(
            tools=[tool],
            messages=[{"role": "user", "content": "hi"}],
        )
        agent.max_turns = 5
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        responses = [
            AgentResponse(
                content=[],
                tool_calls=[ToolCall(id="1", name="read_file",
                                     input={"reason": "x"})],
                stop_reason="tool_use",
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
            AgentResponse(
                content=[],
                tool_calls=[ToolCall(id="r1", name="send_response",
                                     input={"response": "done"})],
                stop_reason="tool_use",
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
        ]
        idx = [0]

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            r = responses[idx[0]]
            idx[0] += 1
            return r

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)

        await run_loop(agent)
        assert idx[0] == 2  # 2 turns
        assert tool.execute_calls == [{}]  # reason stripped

    async def test_max_turns_exhausted(self, monkeypatch) -> None:
        tool = _FakeTool("read_file")
        agent = _mock_agent(
            tools=[tool],
            messages=[{"role": "user", "content": "hi"}],
        )
        agent.max_turns = 2
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[],
            tool_calls=[ToolCall(id="1", name="read_file", input={})],
            stop_reason="tool_use",
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        await run_loop(agent)
        # Last message is the synthetic assistant notice
        last = agent.messages[-1]
        assert last["role"] == "assistant"
        assert "max_turns" in last["content"][0]["text"]

    async def test_pruning_triggers_at_threshold(self, monkeypatch) -> None:
        tool = _FakeTool("read_file", execute_result="A" * 1000)
        agent = _mock_agent(tools=[tool], max_context_tokens=100, prune_context_threshold=0.5)
        # Pre-populate old tool results we'd like to see pruned
        from python_agent.messages import user_message
        agent.messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "old", "name": "read_file", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "old", "content": "X" * 500},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "mid", "name": "read_file", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "mid", "content": "Y" * 500},
            ]},
        ]
        agent.max_turns = 1
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        # input_tokens high enough to exceed threshold (100 * 0.5 = 50)
        response = AgentResponse(
            content=[],
            tool_calls=[ToolCall(id="new", name="read_file", input={})],
            stop_reason="tool_use",
            input_tokens=80, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        await run_loop(agent)
        # Old tool result content should be cleared
        assert agent.messages[2]["content"][0]["content"] == "[Tool output cleared]"

    async def test_pruning_triggers_but_nothing_to_prune(
        self, monkeypatch,
    ) -> None:
        """When threshold is crossed but no tool results to clear, log nothing."""
        tool = _FakeTool("read_file")
        agent = _mock_agent(
            tools=[tool], max_context_tokens=10, prune_context_threshold=0.1,
            messages=[{"role": "user", "content": "hi"}],
        )
        agent.max_turns = 1
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[],
            tool_calls=[ToolCall(id="new", name="read_file", input={})],
            stop_reason="tool_use",
            input_tokens=999, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        # prune returns 0 (nothing to prune)
        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        monkeypatch.setattr(
            "python_agent.processor.prune_tool_results",
            lambda *args, **kwargs: 0,
        )
        await run_loop(agent)
        # Loop completed without error

    async def test_pruning_error_does_not_crash(self, monkeypatch) -> None:
        tool = _FakeTool("read_file")
        agent = _mock_agent(
            tools=[tool], max_context_tokens=10, prune_context_threshold=0.1,
            messages=[{"role": "user", "content": "hi"}],
        )
        agent.max_turns = 1
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[],
            tool_calls=[ToolCall(id="new", name="read_file", input={})],
            stop_reason="tool_use",
            input_tokens=999, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        def broken_prune(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("prune failed")

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        monkeypatch.setattr(
            "python_agent.processor.prune_tool_results", broken_prune,
        )
        # Should not raise
        await run_loop(agent)

    async def test_exception_during_execution_stubs_results(
        self, monkeypatch,
    ) -> None:
        """If _execute_tool_calls raises, the finally stubs tool_results."""
        agent = _mock_agent(messages=[{"role": "user", "content": "hi"}])
        agent.max_turns = 1
        agent.client = MagicMock()
        agent.model = "m"
        agent.env_info = "env"
        agent.api_tools = []

        response = AgentResponse(
            content=[],
            tool_calls=[ToolCall(id="1", name="read_file", input={})],
            stop_reason="tool_use",
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        async def fake_stream(**kwargs):  # type: ignore[no-untyped-def]
            return response

        async def raising_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt()

        monkeypatch.setattr("python_agent.processor.stream_response", fake_stream)
        monkeypatch.setattr(
            "python_agent.processor._execute_tool_calls", raising_exec,
        )

        with pytest.raises(KeyboardInterrupt):
            await run_loop(agent)

        # tool_results stub was appended before the exception propagated
        last = agent.messages[-1]
        assert last["role"] == "user"
        assert "Interrupted" in last["content"][0]["content"]
