"""Environment info adapter — gathers runtime context via OS/subprocess.

Called once per session at the composition root. The returned string
is injected into ``Agent(env_info=...)`` and then into every turn's
system prompt via ``build_system_prompt(env_info=...)``.

Separated from ``core/prompts.py`` so the core has zero infrastructure
imports. The web deployment injects its own ``env_info`` string
(session/user context) without importing this module.
"""

import os
import platform
import subprocess
from datetime import datetime, timezone


def environment_info() -> str:
    """Gather runtime environment context (cwd, git info, platform, shell, date)."""
    cwd = os.getcwd()
    system = platform.system().lower()
    shell = os.environ.get("SHELL", "unknown")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    git_info = _get_git_info()

    git_lines = ""
    if git_info:
        git_lines = (
            f"- **Git branch:** {git_info['branch']}\n"
            f"- **Git status:** {git_info['status']}\n"
        )

    return f"""## Environment
You have been invoked in the following environment:

- **Working directory:** {cwd}
- **Is a git repository:** {git_info is not None}
{git_lines}- **Platform:** {system}
- **Shell:** {shell}
- **Date:** {now}
- **Python interpreter:** not pre-detected; agent must check before first use."""


def _get_git_info() -> dict[str, str] | None:
    """Collect git branch and status. Returns None if not in a git repo."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if branch.returncode != 0:
            return None

        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_text = status.stdout.strip()

        return {
            "branch": branch.stdout.strip(),
            "status": status_text if status_text else "clean",
        }

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
