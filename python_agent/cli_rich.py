"""Rich-based CLI REPL — themed rendering built on the Rich library.

Parallel to ``cli.py`` but adds:

- one ``Theme`` as the single source for style (replacing ~12 ANSI
  constants and a composite ``THINKING_STYLE``),
- a ``thinking .`` / ``.. `` / ``...`` dots spinner (``_Dots`` renderable
  inside a transient ``Live``) — started from three sites: the REPL
  during the pre-stream API wait, ``on_tool_start`` for auto-path
  tools, and ``ask_permission`` after approval for permission-required
  tools — closed by ``close_active`` when the next callback fires,
- ``rich.markdown.Markdown`` rendering of the thinking and response
  streams (streamed live as plain styled text, re-rendered once as
  Markdown when the stream ends so headers, code blocks, lists and
  emphasis get proper formatting without paying a re-parse cost per
  token),
- a header ``Panel`` instead of a hand-drawn box,
- native display-width measurement (fixes the ``_fit`` emoji/CJK bug
  from classic).

``RichOutput`` holds at most one dynamic region at a time — thinking
stream, text stream, or tool spinner. ``close_active`` is the single
point that tears the active region down before the next block.
"""

import readline  # noqa: F401 — enables arrow keys + history in input()
import sys
import time
from typing import Any, Literal

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from . import logging_config
from .agent import Agent, AgentCallbacks
from .client import make_client
from .config import Settings
from .tool_summaries import INTERCEPTED_TOOLS, summarize_params, tool_header_label
from .tools import get_default_tools


# ───────────────────────────────────────────────────────────────────────────
# Theme — single source of truth for style
# ───────────────────────────────────────────────────────────────────────────

THEME = Theme(
    {
        "header": "bold cyan",
        "separator": "color(245)",
        "user.label": "bold blue",
        "user.text": "color(111)",
        "agent.label": "bold green",
        "agent.text": "color(114)",
        "thinking": "dim italic color(141)",
        "tool.name": "bold color(208)",
        "tool.ok": "bold green",
        "tool.fail": "bold red",
        "hint": "color(245)",
        "prompt.question": "bold blue",
        "prompt.choices": "bold red",
        "spinner": "red",
        # Strip the bold that Rich's default Markdown applies to headings
        # (``## Thinking``, ``## Roadmap`` etc.) and keep them in the same
        # dim-italic purple as the thinking body, so a heading doesn't
        # flip into bold + different color mid-stream. Agent-response
        # markdown headings (rare) also take this style — acceptable.
        "markdown.h1": "dim italic color(141)",
        "markdown.h2": "dim italic color(141)",
        "markdown.h3": "dim italic color(141)",
        "markdown.h4": "dim italic color(141)",
    }
)

# Rich's ``Console.input`` resets style after the prompt, so the ANSI
# that colors the typed user input is emitted directly. Must mirror the
# matching ``user.text`` theme entry.
_USER_TEXT_ANSI = "\033[38;5;111m"
_ANSI_RESET = "\033[0m"

# Type of the currently active dynamic region. Either a streaming block
# or a tool spinner — never more than one at a time.
_Kind = Literal["thinking", "text", "spinner"]

# Theme style name used to render each streaming block — both live
# (as ``Text(style=...)``) and in the final ``Markdown`` swap
# (as ``Markdown(..., style=...)``).
_STREAM_STYLES: dict[_Kind, str] = {
    "thinking": "thinking",
    "text": "agent.text",
}

# Streaming blocks rendered indented under the ``agent →`` label.
# The assistant's final response (``text``) stays flush-left so long
# markdown blocks (code, tables, lists) have the full terminal width.
_INDENTED_STREAMS: frozenset[_Kind] = frozenset({"thinking"})

# Left-pad applied to every block of agent activity (tool headers, tool
# results, thinking, assistant text) so the conversation visually nests
# under the ``agent →`` label. The label itself and the turn separator
# rule stay flush-left.
_INDENT = 2


def _indent(renderable: RenderableType) -> Padding:
    """Wrap a renderable with the standard left indent.

    ``expand=False`` avoids filling the line to terminal width with
    trailing spaces — otherwise a line that exactly matches the terminal
    width triggers the terminal's phantom-wrap behavior and swallows the
    next blank ``\\n`` we emit.
    """
    return Padding(renderable, (0, 0, 0, _INDENT), expand=False)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _flush_stdin() -> None:
    """Discard buffered stdin so a just-opened prompt doesn't consume stale Enters.

    Without this, keypresses queued while the agent was thinking get
    consumed by the next ``input()`` and return ``""`` — silently
    approving a permission the user never actually granted.
    """
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError):
        pass  # Windows (no termios) or non-TTY stdin


class _Dots:
    """Renderable that animates ``{label} .`` → ``{label} ..`` → ``{label} ...`` cycling.

    Rich calls ``__rich__`` on every ``Live`` refresh, so time-based
    frame selection gives a clean animation without needing a custom
    entry in Rich's spinner registry. Rendered in the ``spinner``
    theme style so it reads as chrome, not content.
    """

    _CADENCE_S = 0.4  # seconds per dot step

    def __init__(self, label: str) -> None:
        self._label = label
        self._t0 = time.monotonic()

    def __rich__(self) -> Text:
        step = int((time.monotonic() - self._t0) / self._CADENCE_S) % 3 + 1
        return Text(f"{self._label} {'.' * step}", style="spinner")


def _tool_header(name: str, params: dict[str, Any]) -> Text:
    """Render a tool header as ``tool → {name}: {label}`` styled.

    Label text comes from the shared ``tool_header_label`` helper so
    this CLI stays in sync with the classic and TUI variants. The
    whole line uses ``tool.name`` style so it reads as one visual unit.
    """
    label = tool_header_label(name, params)
    text = f"tool → {name}: {label}" if label else f"tool → {name}"
    return Text(text, style="tool.name")


def _tool_detail(name: str, params: dict[str, Any]) -> Text | None:
    """Return the param summary line if it adds info beyond the header.

    When ``reason`` is present the header shows intent; the summary
    (``command``, ``file_path``, ``pattern``…) goes on a second line
    in muted ``hint`` style so the user can see the actual payload.
    When there is no reason, the header already shows the summary and
    this returns ``None`` to avoid a duplicate line.
    """
    if not params.get("reason"):
        return None
    summary = summarize_params(name, params)
    if not summary:
        return None
    return Text(summary, style="hint")


# ───────────────────────────────────────────────────────────────────────────
# Callbacks
# ───────────────────────────────────────────────────────────────────────────


class RichOutput:
    """Agent callbacks that render through a single themed Rich ``Console``."""

    def __init__(self) -> None:
        self.console = Console(theme=THEME, highlight=False)
        self._active: Live | None = None
        self._kind: _Kind | None = None
        self._buf: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────

    def _blank(self) -> None:
        """Emit one blank line of visual breathing room."""
        self.console.print()

    def close_active(self) -> None:
        """Close whatever dynamic region is running and clear state.

        For a streaming block (thinking or text), swap the streamed
        ``Text`` for a one-shot ``Markdown`` render — styled with the
        matching theme entry so color is preserved — then stop the
        ``Live``. For a spinner, just stop it.
        """
        if self._active is None:
            return
        if (
            self._kind in _STREAM_STYLES
            and isinstance(self._active, Live)
            and self._buf
        ):
            md = Markdown(self._buf, style=_STREAM_STYLES[self._kind])
            self._active.update(
                _indent(md) if self._kind in _INDENTED_STREAMS else md,
                refresh=True,
            )
        self._active.stop()
        self._active = None
        self._kind = None
        self._buf = ""

    def start_spinner(self, label: str = "running") -> None:
        """Open a flush-left ``label . .. ...`` spinner.

        Called from tool callbacks with the default ``running`` label and
        from the REPL with ``thinking`` during the pre-stream API wait.
        The next callback closes it via ``close_active``. Transient so
        the spinner cleans up when the next block prints.
        """
        self._active = Live(
            _Dots(label),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._active.start()
        self._kind = "spinner"

    # ── streaming ─────────────────────────────────────────────────────

    def _start_stream(self, kind: _Kind) -> None:
        self._buf = ""
        self._active = Live(
            self._render_stream("", kind),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._active.start()
        self._kind = kind

    def _append(self, delta: str) -> None:
        assert isinstance(self._active, Live) and self._kind in _STREAM_STYLES
        self._buf += delta
        self._active.update(self._render_stream(self._buf, self._kind))

    @staticmethod
    def _render_stream(text: str, kind: _Kind) -> RenderableType:
        body = Text(text, style=_STREAM_STYLES[kind])
        return _indent(body) if kind in _INDENTED_STREAMS else body

    # ── callbacks ─────────────────────────────────────────────────────

    def on_thinking(self, delta: str) -> None:
        """Stream a thinking delta. First delta closes any prior block,
        emits a blank-line separator, and opens a new thinking ``Live``;
        subsequent deltas append to the existing buffer."""
        if self._kind != "thinking":
            self.close_active()
            self._blank()
            self._start_stream("thinking")
        self._append(delta)

    def on_text(self, delta: str) -> None:
        """Stream an assistant-response delta. First delta closes the
        prior block, prints the themed ``agent →`` label, and opens a
        new text ``Live`` that swaps to ``Markdown`` at stream end."""
        if self._kind != "text":
            self.close_active()
            self._blank()
            self.console.print("agent →", style="agent.label")
            self._blank()
            self._start_stream("text")
        self._append(delta)

    def on_tool_start(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> None:
        """Render the tool header + optional detail, then start a spinner
        for non-intercepted tools. ``tool_use_id`` is dropped — the rich
        CLI pairs results positionally with the most recent header."""
        del tool_use_id  # rich pairs positionally with the next result
        self.close_active()
        self._print_tool_preamble(name, params)
        if name not in INTERCEPTED_TOOLS:
            self.start_spinner()

    def _print_tool_preamble(self, name: str, params: dict[str, Any]) -> None:
        """Emit the header line and optional param-detail line for a tool call.

        Shared by ``on_tool_start`` (auto-path) and ``ask_permission``
        (approve-path) — both need to announce the tool before whatever
        comes next (spinner, result, permission prompt).
        """
        self._blank()
        self.console.print(_indent(_tool_header(name, params)))
        detail = _tool_detail(name, params)
        if detail is not None:
            self._blank()
            self.console.print(_indent(detail))

    def on_tool_result(
        self,
        tool_use_id: str,
        name: str,
        params: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        """Close the active spinner/stream and print a ``success ✓`` or
        ``error: … ✗`` line. Pairing is positional — this line always
        belongs to the most recent ``tool → …`` header above it."""
        del tool_use_id, name, params  # rich pairs positionally with the most recent header
        self.close_active()
        self._blank()
        if is_error:
            first_line = result.splitlines()[0] if result else ""
            line = Text(f"error: {first_line} ✗", style="tool.fail")
        else:
            line = Text("success ✓", style="tool.ok")
        self.console.print(_indent(line))

    async def ask_permission(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> bool:
        """Print the tool preamble then block on the ``[Y/n]`` prompt.

        On approval, starts a spinner so the user sees progress while
        the tool runs. The processor skips ``on_tool_start`` for
        permission-required tools, which is why the spinner starts
        here rather than in ``on_tool_start``.
        """
        del tool_use_id  # rich doesn't track tools across the contract
        self.close_active()
        self._print_tool_preamble(name, params)
        self._blank()
        _flush_stdin()
        try:
            answer = self.console.input(
                " " * _INDENT
                + "[prompt.question]Allow execution?[/] "
                "[prompt.choices]\\[Y/n][/] "
            ).strip().lower()
        except EOFError:
            return False
        approved = answer in ("", "y", "yes")
        if approved:
            # The processor skips ``on_tool_start`` for permission-required
            # tools (the header was already rendered above), so start the
            # spinner here instead. The next callback closes it.
            self.start_spinner()
        return approved


# ───────────────────────────────────────────────────────────────────────────
# REPL
# ───────────────────────────────────────────────────────────────────────────


def _print_header(console: Console, model: str, provider: str) -> None:
    body = Text()
    body.append("🤖 Python Agent\n", style="header")
    body.append(f"model: {model}  ·  provider: {provider}\n", style="hint")
    body.append("Type 'exit' or 'quit' to leave.", style="hint")
    console.print()
    console.print(Panel(body, border_style="header", padding=(1, 2)))
    console.print()


def _read_user_input(console: Console) -> str:
    """Read a line from stdin with a themed prompt and colored typed text.

    ``Console.input`` resets styling after the prompt, so the typed
    input would otherwise render in terminal default. Writing the ANSI
    directly and resetting in a ``finally`` keeps the color around the
    user's typing regardless of how ``input()`` returns.
    """
    console.print("user →", style="user.label", end=" ")
    sys.stdout.write(_USER_TEXT_ANSI)
    sys.stdout.flush()
    try:
        return input()
    finally:
        sys.stdout.write(_ANSI_RESET)
        sys.stdout.flush()


async def repl(settings: Settings) -> None:
    """Run the Rich-based interactive REPL."""
    logging_config.configure(debug=settings.debug, log_dir=settings.log_dir)

    output = RichOutput()
    console = output.console

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

    _print_header(console, settings.model, settings.api_provider)

    while True:
        try:
            user_input = _read_user_input(console)
        except (EOFError, KeyboardInterrupt):
            console.print("Goodbye. 👋", style="hint")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            console.print("Goodbye. 👋", style="hint")
            break

        if not user_input.strip():
            continue

        output.start_spinner("thinking")
        try:
            await agent.process_input(user_input)
        except KeyboardInterrupt:
            console.print("(interrupted)", style="hint")
        except Exception as e:
            console.print(f"Error: {e}", style="tool.fail")
        finally:
            output.close_active()

        console.print()
        console.rule(style="separator")
        console.print()
