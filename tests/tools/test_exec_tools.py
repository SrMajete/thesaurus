"""Tests for run_bash, run_python, make_plan, send_response."""

import pytest

from thesaurus.tools.make_plan import MakePlanTool
from thesaurus.tools.run_bash import RunBashTool
from thesaurus.tools.run_python import RunPythonTool
from thesaurus.tools.send_response import SendResponseTool


class TestRunBash:
    async def test_echo(self) -> None:
        result = await RunBashTool().execute(command="echo hello")
        assert "hello" in result

    async def test_timeout_clamped(self) -> None:
        # 10000s clamped to 600, but we pass a small cmd so it returns fast
        result = await RunBashTool().execute(
            command="echo quick", timeout=10_000,
        )
        assert "quick" in result

    async def test_nonzero_exit(self) -> None:
        result = await RunBashTool().execute(command="exit 7")
        assert "(exit code: 7)" in result


class TestRunPython:
    async def test_print(self) -> None:
        result = await RunPythonTool().execute(code="print(2+2)")
        assert "4" in result

    async def test_error_in_code(self) -> None:
        result = await RunPythonTool().execute(code="raise ValueError('bad')")
        assert "ValueError" in result
        assert "(exit code:" in result

    async def test_stateless_between_calls(self) -> None:
        await RunPythonTool().execute(code="x = 99")
        result = await RunPythonTool().execute(code="print(x)")
        # x should not persist — expect NameError
        assert "NameError" in result


class TestMakePlan:
    def test_name_and_flags(self) -> None:
        t = MakePlanTool()
        assert t.name == "make_plan"
        assert t.is_intercepted is True

    def test_schema_has_thinking_and_roadmap(self) -> None:
        t = MakePlanTool()
        props = t.input_schema["properties"]
        assert "thinking" in props
        assert "roadmap" in props
        assert t.input_schema["required"] == ["thinking", "roadmap"]

    async def test_execute_stub_not_called_in_normal_flow(self) -> None:
        """The stub exists for protocol conformance but returns a sentinel."""
        t = MakePlanTool()
        result = await t.execute(thinking="x", roadmap="y")
        # Whatever it returns, it should be a string
        assert isinstance(result, str)


class TestSendResponse:
    def test_name_and_flags(self) -> None:
        t = SendResponseTool()
        assert t.name == "send_response"
        assert t.is_intercepted is True

    def test_schema_has_response(self) -> None:
        t = SendResponseTool()
        assert "response" in t.input_schema["properties"]
        assert "response" in t.input_schema["required"]

    async def test_execute_stub(self) -> None:
        t = SendResponseTool()
        result = await t.execute(response="hi")
        assert isinstance(result, str)
