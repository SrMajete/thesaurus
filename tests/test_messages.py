"""Tests for message construction helpers."""

from python_agent.messages import assistant_message, tool_result, user_message


class TestUserMessage:
    def test_string_content(self) -> None:
        msg = user_message("hello")
        assert msg == {"role": "user", "content": "hello"}

    def test_list_content(self) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
        msg = user_message(blocks)
        assert msg == {"role": "user", "content": blocks}


class TestAssistantMessage:
    def test_content_blocks(self) -> None:
        blocks = [{"type": "text", "text": "hi"}]
        msg = assistant_message(blocks)
        assert msg == {"role": "assistant", "content": blocks}


class TestToolResult:
    def test_success_result(self) -> None:
        r = tool_result("call_1", "ok")
        assert r == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "ok",
        }

    def test_error_result_includes_flag(self) -> None:
        r = tool_result("call_1", "failed", is_error=True)
        assert r == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "failed",
            "is_error": True,
        }

    def test_success_omits_error_flag(self) -> None:
        r = tool_result("call_1", "ok")
        assert "is_error" not in r
