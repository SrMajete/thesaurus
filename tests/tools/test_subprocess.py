"""Tests for the subprocess helper."""

import sys

import pytest

from thesaurus.tools._subprocess import _format_output, run_subprocess


class TestRunSubprocess:
    async def test_exec_mode_captures_stdout(self) -> None:
        result = await run_subprocess("echo", "hello", timeout=5)
        assert "hello" in result

    async def test_shell_mode_captures_stdout(self) -> None:
        result = await run_subprocess("echo shell-mode", timeout=5, shell=True)
        assert "shell-mode" in result

    async def test_nonzero_exit_code_appended(self) -> None:
        result = await run_subprocess("false", timeout=5)
        assert "(exit code: 1)" in result

    async def test_stderr_included(self) -> None:
        result = await run_subprocess(
            sys.executable, "-c", "import sys; sys.stderr.write('oops')",
            timeout=5,
        )
        assert "oops" in result

    async def test_timeout_returns_error(self) -> None:
        result = await run_subprocess("sleep 3", timeout=1, shell=True)
        assert "timed out" in result
        assert "1 seconds" in result

    async def test_nonexistent_executable_returns_error(self) -> None:
        result = await run_subprocess("/definitely/not/a/real/binary", timeout=5)
        assert result.startswith("Error:")


class TestFormatOutput:
    def test_stdout_only(self) -> None:
        result = _format_output(b"hello", b"", 0)
        assert result == "hello"

    def test_stderr_only(self) -> None:
        result = _format_output(b"", b"oops", 0)
        assert result == "oops"

    def test_stdout_and_stderr_joined_with_newline(self) -> None:
        result = _format_output(b"first", b"second", 0)
        assert result == "first\nsecond"

    def test_stdout_ending_with_newline_not_doubled(self) -> None:
        result = _format_output(b"first\n", b"second", 0)
        assert result == "first\nsecond"

    def test_empty_output_shows_placeholder(self) -> None:
        result = _format_output(b"", b"", 0)
        assert result == "(no output)"

    def test_nonzero_exit_code_appended(self) -> None:
        result = _format_output(b"out", b"", 2)
        assert result == "out\n(exit code: 2)"

    def test_zero_exit_no_code_marker(self) -> None:
        result = _format_output(b"out", b"", 0)
        assert "(exit code" not in result

    def test_long_output_truncated(self) -> None:
        big = b"x" * 100_000
        result = _format_output(big, b"", 0)
        assert "truncated" in result

    def test_invalid_utf8_replaced(self) -> None:
        result = _format_output(b"ok \xff bad", b"", 0)
        # Should not raise; the 0xff byte is replaced
        assert "ok" in result
