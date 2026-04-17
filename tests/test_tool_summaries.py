"""Tests for tool summary helpers and intercepted-tool set."""

from thesaurus.tool_summaries import (
    INTERCEPTED_TOOLS,
    SUMMARIZERS,
    summarize_params,
    tool_header_label,
)
from thesaurus.tools.base import ToolName


class TestSummarizers:
    def test_every_registered_tool_has_summarizer(self) -> None:
        for name in ToolName:
            assert name in SUMMARIZERS, f"Missing summarizer for {name}"

    def test_run_bash(self) -> None:
        assert SUMMARIZERS[ToolName.RUN_BASH]({"command": "ls -la"}) == "ls -la"

    def test_run_python_collapses_whitespace(self) -> None:
        result = SUMMARIZERS[ToolName.RUN_PYTHON]({"code": "x = 1\n\n y = 2"})
        assert result == "x = 1 y = 2"

    def test_read_file(self) -> None:
        assert SUMMARIZERS[ToolName.READ_FILE]({"file_path": "/a/b"}) == "/a/b"

    def test_write_file_shows_byte_count(self) -> None:
        result = SUMMARIZERS[ToolName.WRITE_FILE]({
            "file_path": "/a", "content": "hello",
        })
        assert "/a" in result
        assert "5 chars" in result

    def test_edit_file_shows_size_delta(self) -> None:
        result = SUMMARIZERS[ToolName.EDIT_FILE]({
            "file_path": "/a", "old_text": "abc", "new_text": "xyzq",
        })
        assert "3→4" in result

    def test_glob_files(self) -> None:
        assert SUMMARIZERS[ToolName.GLOB_FILES]({"pattern": "*.py"}) == "*.py"

    def test_grep_files(self) -> None:
        assert SUMMARIZERS[ToolName.GREP_FILES]({"pattern": "foo"}) == "foo"

    def test_make_plan_returns_empty(self) -> None:
        assert SUMMARIZERS[ToolName.MAKE_PLAN]({"thinking": "x"}) == ""

    def test_send_response_returns_empty(self) -> None:
        assert SUMMARIZERS[ToolName.SEND_RESPONSE]({"response": "x"}) == ""

    def test_fetch_url(self) -> None:
        assert SUMMARIZERS[ToolName.FETCH_URL]({"url": "https://x.com"}) == "https://x.com"


class TestSummarizeParams:
    def test_known_tool(self) -> None:
        result = summarize_params("run_bash", {"command": "echo hi"})
        assert result == "echo hi"

    def test_unknown_tool_fallback(self) -> None:
        result = summarize_params("unknown", {"a": "b"})
        assert "a" in result
        assert len(result) <= 100

    def test_unknown_tool_long_params_truncated(self) -> None:
        long = {"x": "A" * 500}
        result = summarize_params("unknown", long)
        assert len(result) <= 100


class TestToolHeaderLabel:
    def test_reason_preferred_over_summary(self) -> None:
        result = tool_header_label(
            "run_bash",
            {"reason": "explain", "command": "ls"},
        )
        assert result == "explain"

    def test_empty_reason_falls_back_to_summary(self) -> None:
        result = tool_header_label(
            "run_bash",
            {"reason": "", "command": "ls"},
        )
        assert result == "ls"

    def test_no_reason_uses_summary(self) -> None:
        assert tool_header_label("run_bash", {"command": "ls"}) == "ls"


class TestInterceptedTools:
    def test_contains_make_plan_and_send_response(self) -> None:
        assert ToolName.MAKE_PLAN in INTERCEPTED_TOOLS
        assert ToolName.SEND_RESPONSE in INTERCEPTED_TOOLS

    def test_excludes_non_intercepted(self) -> None:
        assert ToolName.READ_FILE not in INTERCEPTED_TOOLS
        assert ToolName.RUN_BASH not in INTERCEPTED_TOOLS

    def test_is_frozenset(self) -> None:
        assert isinstance(INTERCEPTED_TOOLS, frozenset)
