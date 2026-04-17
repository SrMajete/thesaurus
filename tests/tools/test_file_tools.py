"""Tests for read_file, write_file, edit_file."""

from pathlib import Path

import pytest

from thesaurus.tools.edit_file import EditFileTool
from thesaurus.tools.read_file import ReadFileTool
from thesaurus.tools.write_file import WriteFileTool


class TestReadFile:
    async def test_reads_file_with_line_numbers(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("one\ntwo\nthree")
        result = await ReadFileTool().execute(file_path=str(f))
        assert "   1\tone" in result
        assert "   2\ttwo" in result
        assert "   3\tthree" in result

    async def test_missing_file(self, tmp_path: Path) -> None:
        result = await ReadFileTool().execute(file_path=str(tmp_path / "x"))
        assert "not found" in result

    async def test_directory_path(self, tmp_path: Path) -> None:
        result = await ReadFileTool().execute(file_path=str(tmp_path))
        assert "not a file" in result

    async def test_binary_file_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "bin"
        f.write_bytes(b"\x00\x01\x02")
        result = await ReadFileTool().execute(file_path=str(f))
        assert "binary" in result

    async def test_offset_and_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        result = await ReadFileTool().execute(
            file_path=str(f), offset=3, limit=2,
        )
        assert "   4\tline3" in result
        assert "   5\tline4" in result
        assert "line0" not in result
        assert "line9" not in result

    async def test_more_lines_indicator(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"l{i}" for i in range(100)))
        result = await ReadFileTool().execute(
            file_path=str(f), offset=0, limit=5,
        )
        assert "more lines" in result

    async def test_permission_error(self, tmp_path: Path) -> None:
        f = tmp_path / "nope.txt"
        f.write_text("secret")
        f.chmod(0o000)
        try:
            result = await ReadFileTool().execute(file_path=str(f))
            # On macOS as root or in some environments this won't fail;
            # tolerate both outcomes
            assert "permission denied" in result or "secret" in result
        finally:
            f.chmod(0o644)

    async def test_invalid_utf8_replaced_not_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "odd.txt"
        f.write_bytes(b"ok \xff text")
        result = await ReadFileTool().execute(file_path=str(f))
        assert "ok" in result  # lenient decode


class TestWriteFile:
    async def test_creates_new_file(self, tmp_path: Path) -> None:
        f = tmp_path / "new.txt"
        result = await WriteFileTool().execute(
            file_path=str(f), content="hello",
        )
        assert "Successfully wrote" in result
        assert f.read_text() == "hello"

    async def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("old")
        await WriteFileTool().execute(file_path=str(f), content="new")
        assert f.read_text() == "new"

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "c.txt"
        await WriteFileTool().execute(file_path=str(f), content="deep")
        assert f.read_text() == "deep"

    async def test_permission_error(self, tmp_path: Path) -> None:
        d = tmp_path / "locked"
        d.mkdir()
        d.chmod(0o000)
        try:
            result = await WriteFileTool().execute(
                file_path=str(d / "x.txt"), content="x",
            )
            assert "permission denied" in result or "Error" in result
        finally:
            d.chmod(0o755)

    async def test_os_error_path_is_a_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "dir"
        d.mkdir()
        result = await WriteFileTool().execute(
            file_path=str(d), content="x",
        )
        assert result.startswith("Error")


class TestEditFile:
    async def test_unique_replacement(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1\ny = 2\nz = 3")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="y = 2", new_text="y = 99",
        )
        assert "Applied edit" in result
        assert f.read_text() == "x = 1\ny = 99\nz = 3"

    async def test_empty_old_text_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hi")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="", new_text="x",
        )
        assert "must not be empty" in result

    async def test_identical_texts_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hi")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="hi", new_text="hi",
        )
        assert "identical" in result

    async def test_missing_file(self, tmp_path: Path) -> None:
        result = await EditFileTool().execute(
            file_path=str(tmp_path / "nope"),
            old_text="a", new_text="b",
        )
        assert "not found" in result

    async def test_directory_path(self, tmp_path: Path) -> None:
        result = await EditFileTool().execute(
            file_path=str(tmp_path),
            old_text="a", new_text="b",
        )
        assert "not a file" in result

    async def test_file_size_over_limit(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.txt"
        f.write_text("abc")
        # Patch the class limit so we can test the branch without a 1GB file
        monkeypatch.setattr(EditFileTool, "_MAX_FILE_SIZE", 2)
        result = await EditFileTool().execute(
            file_path=str(f), old_text="abc", new_text="xyz",
        )
        assert "1 GB limit" in result or "exceeds" in result

    async def test_binary_file_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "bin"
        f.write_bytes(b"\x00abc")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="abc", new_text="xyz",
        )
        assert "binary" in result

    async def test_invalid_utf8_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "odd.txt"
        f.write_bytes(b"ok \xff text")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="ok", new_text="xx",
        )
        assert "UTF-8" in result

    async def test_old_text_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="missing", new_text="x",
        )
        assert "not found" in result

    async def test_multiple_matches_without_replace_all(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("foo\nfoo\nfoo")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="foo", new_text="bar",
        )
        assert "found 3 times" in result
        assert f.read_text() == "foo\nfoo\nfoo"  # unchanged

    async def test_replace_all_many_matches(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("foo\nfoo\nfoo")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="foo", new_text="bar",
            replace_all=True,
        )
        assert "Replaced 3 occurrences" in result
        assert f.read_text() == "bar\nbar\nbar"

    async def test_replace_all_single_match_reports_line(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / "a.py"
        f.write_text("x\nfoo\nz")
        result = await EditFileTool().execute(
            file_path=str(f), old_text="foo", new_text="bar",
            replace_all=True,
        )
        assert "(line 2)" in result

    async def test_permission_error_on_write(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello")
        f.chmod(0o444)  # read-only
        try:
            result = await EditFileTool().execute(
                file_path=str(f), old_text="hello", new_text="world",
            )
            assert "permission denied" in result or "Error" in result
        finally:
            f.chmod(0o644)

    async def test_permission_error_on_read(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Covers the read_bytes PermissionError branch."""
        f = tmp_path / "a.txt"
        f.write_text("hello")

        from pathlib import Path as PathClass
        real_read_bytes = PathClass.read_bytes

        def raising(self):  # type: ignore[no-untyped-def]
            if self == f:
                raise PermissionError("denied")
            return real_read_bytes(self)

        monkeypatch.setattr(PathClass, "read_bytes", raising)
        result = await EditFileTool().execute(
            file_path=str(f), old_text="hello", new_text="world",
        )
        assert "permission denied" in result

    async def test_os_error_on_write(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Covers the OSError branch on write_text."""
        f = tmp_path / "a.txt"
        f.write_text("hello")

        from pathlib import Path as PathClass
        real_write_text = PathClass.write_text

        def raising(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self == f:
                raise OSError("disk full")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(PathClass, "write_text", raising)
        result = await EditFileTool().execute(
            file_path=str(f), old_text="hello", new_text="world",
        )
        assert "disk full" in result
