"""Logging configuration.

Sets up three independent, non-overlapping logging channels:

1. **stderr** — terminal output. WARNING+ only (Textual owns the
   screen during a TUI session, so DEBUG on stderr would be invisible
   anyway; warnings and errors surface if the TUI fails to initialize).

2. **agent_log_{timestamp}.log** — INFO, WARNING, ERROR.
   The story: turns, tool calls, permissions, tokens, timing, errors.

3. **agent_debug_{timestamp}.log** — DEBUG only.
   The raw data: full message payloads, API responses, tool results.

The two files are complementary, not overlapping. Read log for the
flow, debug for the payloads. No duplicated lines across files.

Third-party loggers (httpx, httpcore, anthropic SDK) are silenced
to WARNING on all channels.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path


class _InfoAndAboveFilter(logging.Filter):
    """Only allow INFO, WARNING, ERROR, CRITICAL — excludes DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO


class _DebugOnlyFilter(logging.Filter):
    """Only allow DEBUG level — excludes everything else."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.DEBUG


def configure(log_dir: Path) -> None:
    """Initialize logging. Call once at startup."""
    our_logger = logging.getLogger("python_agent")
    our_logger.setLevel(logging.DEBUG)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s\n%(message)s\n")

    # ── stderr (terminal) ─────────────────────────────────────────
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(
        "\n┌ %(asctime)s [%(name)s] %(levelname)s\n└ %(message)s\n"
    ))
    our_logger.addHandler(stderr_handler)

    # ── agent_log (the flow) ──────────────────────────────────────
    log_handler = logging.FileHandler(
        log_dir / f"agent_log_{timestamp}.log", mode="w", encoding="utf-8"
    )
    log_handler.setFormatter(fmt)
    log_handler.addFilter(_InfoAndAboveFilter())
    our_logger.addHandler(log_handler)

    # ── agent_debug (the data) ────────────────────────────────────
    debug_handler = logging.FileHandler(
        log_dir / f"agent_debug_{timestamp}.log", mode="w", encoding="utf-8"
    )
    debug_handler.setFormatter(fmt)
    debug_handler.addFilter(_DebugOnlyFilter())
    our_logger.addHandler(debug_handler)

    # ── Silence third-party loggers ───────────────────────────────
    for name in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
