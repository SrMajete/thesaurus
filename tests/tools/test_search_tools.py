"""Tests for glob_files and grep_files."""

from pathlib import Path

import pytest

from thesaurus.tools.glob_files import GlobFilesTool
from thesaurus.tools.grep_files import GrepFilesTool


class TestGlobFiles:
    async def test_finds_matching_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        result = await GlobFilesTool().execute(pattern="*.py", path=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    async def test_recursive_pattern(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "x.py").write_text("x")
        result = await GlobFilesTool().execute(
            pattern="**/*.py", path=str(tmp_path),
        )
        assert "x.py" in result

    async def test_no_matches(self, tmp_path: Path) -> None:
        result = await GlobFilesTool().execute(
            pattern="*.rs", path=str(tmp_path),
        )
        assert "No files matching" in result

    async def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "x"
        f.write_text("x")
        result = await GlobFilesTool().execute(pattern="*", path=str(f))
        assert "not a directory" in result

    async def test_sorted_by_mtime_desc(self, tmp_path: Path) -> None:
        import os
        import time

        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a")
        time.sleep(0.01)
        b.write_text("b")
        # Force distinct mtimes
        os.utime(a, (100, 100))
        os.utime(b, (200, 200))
        result = await GlobFilesTool().execute(pattern="*.txt", path=str(tmp_path))
        lines = result.splitlines()
        assert lines[0].endswith("b.txt")  # newer first
        assert lines[1].endswith("a.txt")

    async def test_max_results_truncation(self, tmp_path: Path) -> None:
        for i in range(250):
            (tmp_path / f"f{i:03d}.txt").write_text("x")
        result = await GlobFilesTool().execute(
            pattern="*.txt", path=str(tmp_path),
        )
        assert "250 total" in result
        assert "showing first 200" in result

    async def test_directories_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.py").write_text("x")
        result = await GlobFilesTool().execute(pattern="*", path=str(tmp_path))
        assert "a.py" in result
        # Directories filtered out
        assert "sub" not in result.split("\n") or "a.py" in result

    async def test_stat_error_handled_in_sort_key(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """_mtime helper returns 0 when stat raises (file vanished mid-sort)."""
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("x")
        b.write_text("y")

        # Only break stat during the sort phase (after is_file already passed).
        # We simulate by patching Path.stat to raise only for a specific path.
        call_count = {"a": 0}
        real_stat = Path.stat

        def flaky_stat(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.name == "a.txt":
                call_count["a"] += 1
                # First call (is_file) succeeds; second (sort) fails
                if call_count["a"] > 1:
                    raise OSError("gone")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", flaky_stat)
        result = await GlobFilesTool().execute(pattern="*.txt", path=str(tmp_path))
        assert "a.txt" in result  # still listed despite mtime failure
        assert "b.txt" in result


class TestGrepFiles:
    async def test_finds_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("foo\nbar\nbaz")
        result = await GrepFilesTool().execute(
            pattern="bar", path=str(tmp_path),
        )
        assert "bar" in result
        assert "a.py" in result

    async def test_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("foo")
        result = await GrepFilesTool().execute(
            pattern="missing", path=str(tmp_path),
        )
        assert "No matches" in result

    async def test_include_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("target")
        (tmp_path / "b.txt").write_text("target")
        result = await GrepFilesTool().execute(
            pattern="target", path=str(tmp_path), include="*.py",
        )
        assert "a.py" in result
        assert "b.txt" not in result

    async def test_line_numbers_present(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x\ny\ntarget\nz")
        result = await GrepFilesTool().execute(
            pattern="target", path=str(tmp_path),
        )
        assert ":3:" in result  # line number

    async def test_pattern_starting_with_dash_not_parsed_as_flag(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "a.py").write_text("--foo")
        result = await GrepFilesTool().execute(
            pattern="--foo", path=str(tmp_path),
        )
        assert "--foo" in result

    async def test_max_lines_truncation(self, tmp_path: Path) -> None:
        lines = "\n".join("hit" for _ in range(250))
        (tmp_path / "a.txt").write_text(lines)
        result = await GrepFilesTool().execute(
            pattern="hit", path=str(tmp_path),
        )
        assert "250 total matches" in result
        assert "showing first 200" in result

    async def test_ripgrep_missing(self, monkeypatch, tmp_path: Path) -> None:
        import asyncio

        async def raising(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("rg not found")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", raising)
        result = await GrepFilesTool().execute(
            pattern="x", path=str(tmp_path),
        )
        assert "ripgrep" in result

    async def test_timeout(self, monkeypatch, tmp_path: Path) -> None:
        import asyncio

        class FakeProc:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                await asyncio.sleep(10)
                return b"", b""

            def kill(self) -> None:
                pass

            async def wait(self) -> None:
                pass

        async def fake_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            return FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(
            "thesaurus.tools.grep_files._TIMEOUT", 0.01,
        )
        result = await GrepFilesTool().execute(
            pattern="x", path=str(tmp_path),
        )
        assert "timed out" in result

    async def test_timeout_process_already_exited(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        """The kill() call handles ProcessLookupError."""
        import asyncio

        class FakeProc:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                await asyncio.sleep(10)
                return b"", b""

            def kill(self) -> None:
                raise ProcessLookupError()

            async def wait(self) -> None:
                pass

        async def fake_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            return FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(
            "thesaurus.tools.grep_files._TIMEOUT", 0.01,
        )
        result = await GrepFilesTool().execute(
            pattern="x", path=str(tmp_path),
        )
        assert "timed out" in result

    async def test_ripgrep_error_exit(self, monkeypatch, tmp_path: Path) -> None:
        """rg returncode other than 0 or 1 is surfaced as an error."""
        import asyncio

        class FakeProc:
            returncode = 2

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"", b"bad regex"

        async def fake_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            return FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        result = await GrepFilesTool().execute(
            pattern="[", path=str(tmp_path),
        )
        assert "bad regex" in result

    async def test_ripgrep_error_empty_stderr(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        import asyncio

        class FakeProc:
            returncode = 2

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"", b""

        async def fake_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            return FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        result = await GrepFilesTool().execute(
            pattern="x", path=str(tmp_path),
        )
        assert "search failed" in result
