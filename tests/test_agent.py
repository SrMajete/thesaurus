"""Tests for the Agent class and callbacks."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python_agent.agent import Agent, AgentCallbacks


def _noop_callbacks() -> AgentCallbacks:
    return AgentCallbacks(
        on_thinking=lambda t: None,
        on_text=lambda t: None,
        on_tool_start=lambda *a: None,
        on_tool_result=lambda *a: None,
        ask_permission=AsyncMock(return_value=True),
    )


class _FakeTool:
    name = "fake_tool"
    description = "fake"
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, **params: Any) -> str:
        return "ok"


class _InvalidNameTool:
    name = "NotSnakeCase"
    description = "bad"
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **params: Any) -> str:
        return "ok"


class _SingleWordTool:
    name = "oneword"  # no underscore
    description = "bad"
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **params: Any) -> str:
        return "ok"


class TestAgentInit:
    def test_basic_construction(self) -> None:
        a = Agent(
            client=MagicMock(),
            model="claude-test",
            tools=[_FakeTool()],
            callbacks=_noop_callbacks(),
        )
        assert a.model == "claude-test"
        assert a.messages == []
        assert a.plan is None
        assert a.total_input_tokens == 0
        assert a.last_context_tokens == 0
        assert a.max_context_tokens == 200_000
        assert a.prune_context_threshold == 0.8

    def test_rejects_invalid_tool_name(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            Agent(
                client=MagicMock(),
                model="m",
                tools=[_InvalidNameTool()],
                callbacks=_noop_callbacks(),
            )

    def test_rejects_single_word_tool_name(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            Agent(
                client=MagicMock(),
                model="m",
                tools=[_SingleWordTool()],
                callbacks=_noop_callbacks(),
            )

    def test_custom_max_context_and_threshold(self) -> None:
        a = Agent(
            client=MagicMock(),
            model="m",
            tools=[_FakeTool()],
            callbacks=_noop_callbacks(),
            max_context_tokens=1_000_000,
            prune_context_threshold=0.9,
        )
        assert a.max_context_tokens == 1_000_000
        assert a.prune_context_threshold == 0.9

    def test_api_tools_formatted(self) -> None:
        a = Agent(
            client=MagicMock(),
            model="m",
            tools=[_FakeTool()],
            callbacks=_noop_callbacks(),
        )
        # api_tools should include the reason field injected
        assert "reason" in a.api_tools[0]["input_schema"]["properties"]

    def test_env_info_captured(self) -> None:
        a = Agent(
            client=MagicMock(),
            model="m",
            tools=[_FakeTool()],
            callbacks=_noop_callbacks(),
        )
        assert isinstance(a.env_info, str)
        assert len(a.env_info) > 0


class TestProcessInput:
    async def test_appends_user_message_and_invokes_loop(
        self, monkeypatch,
    ) -> None:
        a = Agent(
            client=MagicMock(),
            model="m",
            tools=[_FakeTool()],
            callbacks=_noop_callbacks(),
        )

        ran = {"called": False}

        async def fake_loop(agent: Agent) -> None:
            ran["called"] = True
            assert agent.messages[-1] == {"role": "user", "content": "hi"}

        monkeypatch.setattr("python_agent.agent.run_loop", fake_loop)
        await a.process_input("hi")
        assert ran["called"] is True
        assert a.messages[0]["content"] == "hi"


class TestAgentCallbacks:
    def test_frozen_dataclass(self) -> None:
        cb = _noop_callbacks()
        with pytest.raises(Exception):
            cb.on_thinking = lambda t: None  # type: ignore[misc]

    def test_required_fields(self) -> None:
        """All five callbacks are required at construction."""
        with pytest.raises(TypeError):
            AgentCallbacks(  # type: ignore[call-arg]
                on_thinking=lambda t: None,
                on_text=lambda t: None,
                on_tool_start=lambda *a: None,
                on_tool_result=lambda *a: None,
                # missing ask_permission
            )
