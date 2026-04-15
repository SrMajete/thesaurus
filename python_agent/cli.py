"""CLI REPL — thin I/O bridge between the user and the agent.

All user-facing I/O lives here. The CLI creates an Agent, provides
callback implementations for terminal output, and runs the input loop.
Import readline for arrow-key navigation and command history.
"""

import os
import readline  # noqa: F401 — enables arrow keys + history in input()
import sys
from enum import Enum, auto
from typing import Any

from . import logging_config
from .agent import Agent, AgentCallbacks
from .client import make_client
from .config import Settings
from .tool_summaries import summarize_params, tool_header_label
from .tools import get_default_tools


class _DisplayState(Enum):
    """Tracks what the terminal last rendered so transitions space correctly."""

    NONE = auto()
    THINKING = auto()
    TEXT = auto()
    TOOL = auto()


# ───────────────────────────────────────────────────────────────────────────
# Colors (ANSI escape codes)
# ───────────────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"

ORANGE = "\033[38;5;208m"
PURPLE = "\033[38;5;141m"
GRAY = "\033[38;5;245m"
LIGHT_BLUE = "\033[38;5;111m"
LIGHT_GREEN = "\033[38;5;114m"

THINKING_STYLE = f"{BOLD}{DIM}{ITALIC}{PURPLE}"
SEPARATOR = f"{GRAY}{'─' * 56}{RESET}"


# ───────────────────────────────────────────────────────────────────────────
# Terminal output callbacks
# ───────────────────────────────────────────────────────────────────────────


class TerminalOutput:
    """Implements all agent callbacks for terminal display.

    Groups the callbacks into a class so they can share display state
    for handling transitions between thinking, text, and tool output.
    """

    def __init__(self) -> None:
        self._last: _DisplayState = _DisplayState.NONE
        self._col: int = 0

    @property
    def _width(self) -> int:
        """Terminal width with margin for comfortable reading."""
        try:
            return os.get_terminal_size().columns - 10
        except OSError:
            return 72

    def _print_wrapped(self, text: str, style: str = "") -> None:
        """Print text with word wrapping at terminal width."""
        print(style, end="", flush=True)
        for char in text:
            if char == "\n":
                print(RESET)
                self._col = 0
                print(style, end="", flush=True)
            elif self._col >= self._width and char == " ":
                print(RESET)
                self._col = 0
                print(style, end="", flush=True)
            else:
                print(char, end="", flush=True)
                self._col += 1
        print(RESET, end="", flush=True)

    def _fit(self, text: str) -> str:
        """Truncate text to fit within terminal width."""
        if len(text) <= self._width:
            return text
        return text[:self._width - 1] + "…"

    def on_thinking(self, delta: str) -> None:
        """Print thinking body in dim italic purple, beneath plan's tool line.

        Called from inside the streaming loop while plan.thinking / plan.roadmap
        are being generated. The tool header has just been printed by
        on_tool_start and ends with a trailing space (no newline), so we emit
        a blank-line separation (two newlines) before the body begins.
        """
        if self._last != _DisplayState.THINKING:
            print("\n\n", end="", flush=True)
            self._col = 0
        self._last = _DisplayState.THINKING
        self._print_wrapped(delta, THINKING_STYLE)

    def on_text(self, delta: str) -> None:
        """Print streamed text in cyan.

        Called from inside the streaming loop while respond.response is being
        generated. Same blank-line separation (two newlines) as on_thinking
        when transitioning from a tool header line.
        """
        if self._last == _DisplayState.THINKING:
            print(f"{RESET}\n\n", end="", flush=True)
            self._col = 0
        elif self._last == _DisplayState.TOOL:
            print("\n\n", end="", flush=True)
            self._col = 0
        self._last = _DisplayState.TEXT
        self._print_wrapped(delta, LIGHT_GREEN)

    def on_tool_start(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> None:
        """Print tool name and the model's reason, waiting for result on the same line.

        Leaves the cursor mid-line on the rendered header (with a trailing
        space after ``{RESET}``), so ``on_tool_result`` can complete the
        line with ✓/✗. We update ``self._col`` to the header's visible
        width + 1 for the trailing space — this gives ``on_tool_result``
        a single source of truth for the cursor position, and matches the
        same convention ``_print_wrapped`` uses.
        """
        del tool_use_id  # classic uses positional pairing with the next result
        if self._last == _DisplayState.THINKING:
            print(RESET, end="", flush=True)
        sep = "\n" if self._last == _DisplayState.TOOL else "\n\n"
        self._last = _DisplayState.TOOL
        label = tool_header_label(name, params)
        line = self._fit(f"tool → {name}: {label}" if label else f"tool → {name}")
        print(f"{sep}{BOLD}{ORANGE}{line}{RESET} ", end="", flush=True)
        self._col = len(line) + 1

    def on_tool_result(
        self,
        tool_use_id: str,
        name: str,
        params: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        """Print the tool result on its own line: ``✓ success`` or ``✗ error: {message}``.

        Tool attribution is positional — each result line corresponds to the
        most recent ``on_tool_start`` header, and for batched parallel tools
        the results print in the same order as the headers because
        ``asyncio.gather`` preserves order. The tool name is therefore not
        repeated on the result line; the header above already names it.

        Leading whitespace depends on where the cursor is:

        - Auto path: ``on_tool_start`` left the cursor mid-line on the
          header (``_col > 0``), so we emit ``\\n\\n`` to finish the
          header and leave one blank line before ✓/✗.
        - Approve path: ``ask_permission``'s ``input()`` already consumed
          the user's Enter and the cursor is at column 0 (``_col == 0``),
          so ``\\n`` is enough to leave one blank line.
        """
        del tool_use_id, name, params  # classic pairs positionally with the most recent header
        lead = "\n\n" if self._col > 0 else "\n"
        if is_error:
            error = self._fit(f"error: {result}")
            print(f"{lead}{BOLD}{RED}{error} ✗{RESET}", flush=True)
        else:
            print(f"{lead}{BOLD}{GREEN}success ✓{RESET}", flush=True)
        self._col = 0

    async def ask_permission(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> bool:
        """Prompt the user before executing a non-read-only tool.

        Shows the model's reason AND the raw parameters — the user needs to
        see what they're actually approving, not just a paraphrase.
        """
        del tool_use_id  # classic doesn't track tools across the contract
        if self._last == _DisplayState.THINKING:
            print(RESET, end="", flush=True)
        sep = "\n" if self._last == _DisplayState.TOOL else "\n\n"
        self._last = _DisplayState.TOOL
        self._col = 0

        reason = params.get("reason")
        summary = summarize_params(name, params)

        header = self._fit(f"tool → {name}: {reason}" if reason else f"tool → {name}: {summary}")
        detail = self._fit(summary) if reason else None

        try:
            print(f"{sep}{BOLD}{ORANGE}{header}{RESET}", flush=True)
            if detail:
                print(f"\n{GRAY}{detail}{RESET}", flush=True)
            prompt = f"{BOLD}{BLUE}Allow execution?{RESET} {BOLD}{RED}[Y/n]{RESET} "
            # Discard any buffered stdin (Enter presses queued while the
            # agent was thinking) so the prompt only accepts keypresses
            # typed AFTER it appeared. Without this, queued newlines are
            # consumed by input() and return "" → treated as default "yes"
            # → silent permission grant the user never actually gave.
            try:
                import termios
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except (ImportError, OSError):
                pass  # Windows (no termios) or non-TTY stdin — nothing to flush
            answer = input(f"\n{prompt}").strip().lower()
        except EOFError:
            return False

        return answer in ("", "y", "yes")

# ───────────────────────────────────────────────────────────────────────────
# REPL
# ───────────────────────────────────────────────────────────────────────────


async def repl(settings: Settings) -> None:
    """Run the interactive classic-CLI REPL."""
    logging_config.configure(debug=settings.debug, log_dir=settings.log_dir)

    output = TerminalOutput()

    agent = Agent(
        client=make_client(settings),
        model=settings.model,
        tools=get_default_tools(),
        callbacks=AgentCallbacks(
            on_thinking=output.on_thinking,
            on_text=output.on_text,
            on_tool_start=output.on_tool_start,
            on_tool_result=output.on_tool_result,
            ask_permission=output.ask_permission,
        ),
        max_turns=settings.max_turns,
    )

    # Header (emoji takes 2 columns, so pad accounts for that)
    # Inner box = 56 cols, visible text before model name = 27 cols (emoji = 2).
    # Truncate model names longer than 29 chars to keep the box aligned.
    max_model = 29
    model = settings.model if len(settings.model) <= max_model else settings.model[:max_model - 1] + "…"
    pad = " " * (max_model - len(model))
    B = f"{BOLD}{CYAN}"  # box style
    print()
    print(f"  {B}╔════════════════════════════════════════════════════════╗{RESET}")
    print(f"  {B}║{RESET}                                                        {B}║{RESET}")
    print(f"  {B}║{RESET}   🤖 {BOLD}Python Agent{RESET}  {GRAY}model: {model}{RESET}{pad}{B}║{RESET}")
    print(f"  {B}║{RESET}                                                        {B}║{RESET}")
    print(f"  {B}║{RESET}   {GRAY}Type 'exit' or 'quit' to leave.{RESET}                      {B}║{RESET}")
    print(f"  {B}║{RESET}                                                        {B}║{RESET}")
    print(f"  {B}╚════════════════════════════════════════════════════════╝{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{BOLD}{BLUE}user →{RESET} {LIGHT_BLUE}")
            print(RESET, end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}Goodbye. 👋{RESET}")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            print(f"{GRAY}Goodbye. 👋{RESET}")
            break

        if not user_input.strip():
            continue

        print(f"\n{BOLD}{GREEN}agent →{RESET} ", end="", flush=True)

        try:
            await agent.process_input(user_input)

        except KeyboardInterrupt:
            print(f"\n{GRAY}(interrupted){RESET}")

        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}", file=sys.stderr)

        print(f"\n\n{SEPARATOR}\n")
