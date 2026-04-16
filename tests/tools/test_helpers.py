"""Tests for shared tool helpers."""

from pathlib import Path

import pytest

from python_agent.tools._helpers import (
    MAX_OUTPUT_CHARS,
    TIMEOUT_LIMIT,
    clamp_timeout,
    is_binary,
    truncate,
    validate_file_path,
)


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert truncate("hello") == "hello"

    def test_exact_max_unchanged(self) -> None:
        text = "x" * MAX_OUTPUT_CHARS
        assert truncate(text) == text

    def test_long_text_truncated_with_marker(self) -> None:
        text = "a" * 100_000
        result = truncate(text)
        assert len(result) <= MAX_OUTPUT_CHARS + 100  # overhead for marker
        assert "truncated" in result
        assert "100000" in result  # original length mentioned

    def test_head_and_tail_preserved(self) -> None:
        text = "START" + "x" * 100_000 + "END"
        result = truncate(text)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_custom_max_chars(self) -> None:
        text = "x" * 500
        result = truncate(text, max_chars=100)
        assert len(result) <= 200  # 100 + marker overhead
        assert "truncated" in result

    def test_empty_string(self) -> None:
        assert truncate("") == ""


class TestClampTimeout:
    def test_below_minimum_clamped_to_one(self) -> None:
        assert clamp_timeout(0) == 1
        assert clamp_timeout(-5) == 1

    def test_above_limit_clamped(self) -> None:
        assert clamp_timeout(10_000) == TIMEOUT_LIMIT

    def test_within_range_unchanged(self) -> None:
        assert clamp_timeout(30) == 30
        assert clamp_timeout(1) == 1
        assert clamp_timeout(TIMEOUT_LIMIT) == TIMEOUT_LIMIT


class TestValidateFilePath:
    def test_nonexistent_returns_error(self, tmp_path: Path) -> None:
        err = validate_file_path(tmp_path / "missing.txt")
        assert err is not None
        assert "not found" in err

    def test_directory_returns_error(self, tmp_path: Path) -> None:
        err = validate_file_path(tmp_path)
        assert err is not None
        assert "not a file" in err

    def test_valid_file_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.txt"
        f.write_text("hi")
        assert validate_file_path(f) is None


class TestIsBinary:
    def test_text_is_not_binary(self) -> None:
        assert is_binary(b"hello world") is False

    def test_null_byte_is_binary(self) -> None:
        assert is_binary(b"hello\x00world") is True

    def test_empty_is_not_binary(self) -> None:
        assert is_binary(b"") is False

    def test_null_byte_beyond_check_range_ignored(self) -> None:
        content = b"a" * 9000 + b"\x00"
        assert is_binary(content) is False
        # But with a larger check_bytes, it should find it
        assert is_binary(content, check_bytes=10_000) is True

    def test_unicode_text_is_not_binary(self) -> None:
        assert is_binary("héllo wörld".encode("utf-8")) is False
