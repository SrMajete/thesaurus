"""Tests for system prompt construction and environment info."""

import subprocess
from unittest.mock import MagicMock, patch

from python_agent.prompts import (
    _current_plan_section,
    _get_git_info,
    build_system_prompt,
    environment_info,
)


class TestBuildSystemPrompt:
    def test_returns_list_of_blocks(self) -> None:
        blocks = build_system_prompt(env_info="env stuff")
        assert isinstance(blocks, list)
        assert all(b["type"] == "text" for b in blocks)

    def test_no_plan_single_cached_block(self) -> None:
        blocks = build_system_prompt(env_info="env")
        assert len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_with_plan_two_blocks(self) -> None:
        blocks = build_system_prompt(env_info="env", current_plan="my plan")
        assert len(blocks) == 2
        # Static block is cached
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        # Plan block is not cached
        assert "cache_control" not in blocks[1]
        assert "my plan" in blocks[1]["text"]

    def test_env_info_included_in_static(self) -> None:
        blocks = build_system_prompt(env_info="TEST_ENV_MARKER")
        assert "TEST_ENV_MARKER" in blocks[0]["text"]

    def test_static_includes_identity_and_principles(self) -> None:
        blocks = build_system_prompt(env_info="env")
        text = blocks[0]["text"]
        assert "Identity" in text
        assert "Code Quality" in text or "CODE_QUALITY" in text.upper()


class TestCurrentPlanSection:
    def test_includes_plan_content(self) -> None:
        s = _current_plan_section("step 1\nstep 2")
        assert "step 1" in s
        assert "Current Plan" in s

    def test_instruction_present(self) -> None:
        s = _current_plan_section("x")
        assert "evolve" in s.lower() or "update" in s.lower()


class TestEnvironmentInfo:
    def test_includes_cwd(self) -> None:
        info = environment_info()
        import os
        assert os.getcwd() in info

    def test_includes_platform(self) -> None:
        info = environment_info()
        import platform
        assert platform.system().lower() in info

    def test_includes_date(self) -> None:
        info = environment_info()
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in info

    def test_marks_git_repo_status(self) -> None:
        info = environment_info()
        # Either "Is a git repository: True" or "False"
        assert "git repository" in info

    def test_no_git_info_when_not_a_repo(self, monkeypatch) -> None:
        def fake_git(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = MagicMock()
            result.returncode = 128  # "fatal: not a git repository"
            result.stdout = ""
            return result
        monkeypatch.setattr(subprocess, "run", fake_git)
        info = environment_info()
        assert "False" in info
        assert "Git branch" not in info


class TestGetGitInfo:
    def test_in_git_repo_returns_branch_and_status(self) -> None:
        info = _get_git_info()
        # This test lives inside a git repo, so this should succeed
        assert info is not None
        assert "branch" in info
        assert "status" in info

    def test_not_a_repo_returns_none(self, monkeypatch) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = MagicMock()
            result.returncode = 128
            result.stdout = ""
            return result
        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _get_git_info() is None

    def test_git_not_installed_returns_none(self, monkeypatch) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("git not found")
        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _get_git_info() is None

    def test_timeout_returns_none(self, monkeypatch) -> None:
        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.TimeoutExpired(cmd="git", timeout=5)
        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _get_git_info() is None

    def test_clean_status(self, monkeypatch) -> None:
        calls = [0]

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = MagicMock()
            result.returncode = 0
            if calls[0] == 0:  # branch
                result.stdout = "main\n"
            else:  # status
                result.stdout = ""
            calls[0] += 1
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        info = _get_git_info()
        assert info == {"branch": "main", "status": "clean"}

    def test_dirty_status(self, monkeypatch) -> None:
        calls = [0]

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = MagicMock()
            result.returncode = 0
            if calls[0] == 0:
                result.stdout = "feature\n"
            else:
                result.stdout = " M file.py\n"
            calls[0] += 1
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        info = _get_git_info()
        assert info == {"branch": "feature", "status": "M file.py"}
