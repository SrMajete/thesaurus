"""Textual-based CLI — the agent's only interface.

Uses Textual for a true TUI experience:

- fixed ``Input`` docked at the bottom (never scrolls away)
- scrollable ``VerticalScroll`` message history above
- mouse support, Ctrl+L clear, Ctrl+C/D quit
- collapsible tool-call blocks with inline spinners and result rows
- live-streaming markdown for thinking and assistant response
- inline permission prompts (no modal interruption)

Implements the ``AgentCallbacks`` contract the core expects, so
``agent.py``, ``processor.py``, ``api_client.py``, and the tools are
unchanged. Callbacks fire on the Textual event loop (same asyncio loop
``agent.process_input`` awaits on), so no thread-bridge indirection is
needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pyfiglet import Figlet
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, TextArea

from . import logging_config
from .agent import Agent, AgentCallbacks
from .client import fetch_max_context_tokens, make_client
from .config import Settings
from .tool_summaries import INTERCEPTED_TOOLS, summarize_params, tool_header_label
from .tools import get_default_tools
from .tools.base import ToolName

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Theme — single Textual CSS string
# ───────────────────────────────────────────────────────────────────────────

# Semantic class names bind to ANSI-256 shades. Hex values below are
# the standard xterm 256-color mappings for ANSI color indices:
#   ansi 111 = user.text (light blue)
#   ansi 114 = agent.text (light green)
#   ansi 141 = thinking (purple)
#   ansi 208 = tool.name (orange)
#   ansi 245 = hint / separator (gray)
_CSS = """
/* Transparent Screen + message pane so the terminal's native
   background shows through everywhere — otherwise Textual's themed
   ``$background`` paints a rectangle that doesn't quite match the
   user's terminal colors, making code blocks look inset. */
Screen {
    layers: base modal;
    background: black 0%;
}

#messages {
    height: 1fr;
    padding: 0 1;
    background: black 0%;
}

/* Prompt border mirrors the user/agent text palette: unfocused ->
   ``.user`` cyan (``#5fd7d7``) so the box reads as the user's
   editable space; focused -> ``.agent`` pastel green (``#87d787``)
   signalling "input is hot, ready to send to the agent". */
#prompt {
    dock: bottom;
    border: round #5fd7d7;
    height: auto;
    min-height: 3;
    max-height: 8;
    margin: 0;
    padding: 0 1;
}

/* Cumulative token counter — one line tall. Sits directly above the
   prompt via flow layout: the prompt's ``dock: bottom`` takes it out
   of flow, so this widget naturally lands at the bottom of the
   remaining area. ``margin-top: 1`` leaves a blank row between the
   last scrollback widget (usually a turn separator) and the counter
   so they don't visually collide. ``padding: 0 1`` matches
   ``#messages``'s own padding so the counter aligns with the left
   edge of scrollback content. Colors come from the Rich ``Text``
   inside, so no CSS ``color:`` property here. */
#token-counter {
    height: 2;
    margin-top: 2;
    padding: 0 1;
    text-align: left;
    background: black 0%;
}

#prompt:focus {
    border: round #87d787;
}

.banner {
    color: #00aaaa;
    text-style: bold;
    padding: 1 0 0 1;
    height: auto;
}

/* User message text — bright turquoise cyan. Brighter and cooler
   than the banner's ``#00aaaa`` teal; visually pairs with the agent's
   ``#87d787`` light green at a similar lightness level. Two rows of
   top padding give each user message clear breathing room above —
   single source of truth for "space between prior content and the
   next user turn," whether that prior content is a startup hint
   (first turn) or a turn-separator rule (subsequent turns). */
.user {
    color: #5fd7d7;
    padding: 2 0 0 0;
}

.hint {
    color: #8a8a8a;
    text-style: italic;
}

/* .tool-header has no rule: color + weight come from the inline
   ``Text`` returned by ``_tool_header_text``. See that function's
   docstring for the split. */

/* tool-detail / tool-ok / tool-fail are mounted INSIDE ToolCall, so
   their top padding provides the blank between header and child rows.
   ToolCall itself is separated from its siblings via explicit spacer
   widgets emitted by AgentApp._mount_blank.
   No ``color`` here — ``_detail_renderable`` returns either a
   ``Syntax`` renderable (with its own monokai palette) or a ``Text``
   with inline ``style="#8a8a8a"``. Setting a CSS color would cascade
   over the Syntax output and flatten the highlighting. */
.tool-detail {
    padding: 1 0 0 2;
    background: black 0%;
}

.tool-ok {
    color: green;
    text-style: bold;
    padding: 1 0 0 2;
}

.tool-fail {
    color: #ff0000;
    text-style: bold;
    padding: 1 0 0 2;
}

.spacer {
    height: 1;
}

/* Nested one level further right than tool headers so the stream
   reads as content belonging to the enclosing ``make_plan`` block.
   Dim italic purple so thinking recedes visually — it's the agent's
   scratchpad, not the deliverable. Paired with ``dim`` in the
   Markdown ``_BASE_STYLE`` below. */
.thinking {
    color: #af87ff;
    text-style: italic;
    padding: 0 0 0 4;
    height: auto;
}

.agent-label {
    color: #00ff5f;
    text-style: bold;
    height: 1;
}

/* Agent-response stream. Widget ``classes=kind`` where kind is
   ``"agent"`` — selector must match that exact name. */
.agent {
    color: #87d787;
    height: auto;
}

Spinner {
    color: #ff0000;
    padding: 0 0 0 2;
    height: 1;
    margin-top: 1;
}

/* Inline 'Allow execution? [Y/n]' prompt mounted below a pending
   ToolCall when the agent requests a permission-gated tool. Padding
   matches ``.tool-detail`` so it reads as a child row of the tool
   block above. The blue accent is distinct from tool headers (yellow)
   and results (green/red) so the user recognises the row as an
   interactive prompt at a glance. */
.permission-prompt {
    color: #87afff;
    padding: 1 0 0 4;
}
"""


# ASCII-art banner rendered once at startup. ``ansi_shadow`` is pyfiglet's
# chunky 3D-shadow font — best "techy terminal tool" readability at our
# scale. Swap the constant here to try another font (e.g. ``slant``,
# ``big``, ``standard``); list available ones with ``pyfiglet -l``.
_BANNER_TEXT = "PYTHON AGENT"
_BANNER_FONT = "ansi_shadow"


def _render_banner() -> str:
    """Render the startup banner as pre-formatted ASCII art.

    Rendered eagerly once so app startup doesn't pay the figlet cost
    each time the app launches — pyfiglet's font loader parses the
    font file on first call and caches per-process.
    """
    return Figlet(font=_BANNER_FONT).renderText(_BANNER_TEXT).rstrip()


_BANNER = _render_banner()


_CONTEXT_PCT_COLORS: list[tuple[int, str]] = [
    (90, "red"),
    (75, "#ff8700"),
    (60, "yellow"),
    (40, "#5fd7d7"),
    (20, "green"),
    (0, "white"),
]


def _context_pct_color(pct: int) -> str:
    """Return the color for a context usage percentage."""
    for threshold, color in _CONTEXT_PCT_COLORS:
        if pct >= threshold:
            return color
    return "white"


def _scale(peak: int) -> tuple[str, int, int]:
    """Pick unit label, divisor, and decimal places for token display."""
    if peak < 10_000:
        return "", 1, 0
    if peak < 1_000_000:
        return "K", 1_000, 1
    return "M", 1_000_000, 1


def _format_token_line(
    in_tokens: int,
    cached_tokens: int,
    out_tokens: int,
    label: str = "",
    context_tokens: int | None = None,
    max_context_tokens: int | None = None,
) -> Text:
    """Build a colored token info line.

    ``label`` (e.g. ``"delta"`` or ``"cumulative"``) is inserted after
    ``tokens`` in the prefix. Context info is appended when both
    context params are provided.
    """
    suffix, divisor, decimals = _scale(max(in_tokens, cached_tokens, out_tokens))

    def fmt(n: int) -> str:
        return f"{n:,}" if divisor == 1 else f"{n / divisor:.{decimals}f}"

    # Build prefix: tokens(label, K) → or tokens(label) → or tokens(K) →
    if label and suffix:
        prefix = f"tokens({label}, {suffix}) → "
    elif label:
        prefix = f"tokens({label}) → "
    else:
        prefix = f"tokens({suffix}) → " if suffix else "tokens → "

    t = Text()
    t.append(prefix, style="bright_white")
    t.append("input=", style="#5fd7d7")
    t.append(fmt(in_tokens), style="#5fffff")
    t.append(" (", style="white")
    t.append(fmt(cached_tokens), style="white")
    t.append(" cached), ", style="white")
    t.append("output=", style="#87d787")
    t.append(fmt(out_tokens), style="#00ff5f")

    if context_tokens is not None and max_context_tokens and max_context_tokens > 0:
        # Context ratio uses shared scale from the larger value.
        ctx_suffix, ctx_div, ctx_dec = _scale(max_context_tokens)

        def ctx_fmt(n: int) -> str:
            return f"{n:,}" if ctx_div == 1 else f"{n / ctx_div:.{ctx_dec}f}"

        pct = min(100.0, context_tokens / max_context_tokens * 100)
        t.append(", ", style="white")
        t.append("context=", style="bright_white")
        t.append(f"{ctx_fmt(context_tokens)}{ctx_suffix}", style="white")
        t.append("/", style="white")
        t.append(f"{ctx_fmt(max_context_tokens)}{ctx_suffix}", style="white")
        t.append(f" ({pct:.1f}%)", style=_context_pct_color(int(pct)))

    return t


def _code_block(code: str, lexer: str) -> Syntax:
    """Render ``code`` as a monokai-highlighted, word-wrapped ``Syntax`` block.

    ``background_color="default"`` keeps the terminal background
    transparent — foreground token colors from monokai remain, but
    the block blends into the surrounding TUI canvas instead of
    painting its own dark rectangle. Centralised so all tool-detail
    code renderings share the same theme / padding / wrap behavior —
    change it here to restyle every tool at once.
    """
    return Syntax(
        code,
        lexer,
        theme="monokai",
        word_wrap=True,
        padding=(0, 1),
        background_color="default",
    )


def _lexer_for_path(path: str) -> str:
    """Guess a Pygments lexer name from a file extension.

    Falls back to ``"text"`` when the path is empty or unrecognised
    (empty-string lexer would make ``Syntax`` raise).
    """
    if not path:
        return "text"
    try:
        from pygments.lexers import get_lexer_for_filename
        from pygments.util import ClassNotFound
    except ImportError:
        return "text"
    try:
        return get_lexer_for_filename(path).aliases[0]
    except (ClassNotFound, IndexError):
        return "text"


def _detail_renderable(name: str, params: dict[str, Any]) -> RenderableType:
    """Return the tool-detail pane as the richest possible rendering
    of what the tool will actually do.

    Every tool gets a ``Syntax``-framed block so the payload sits in
    the same visual weight class beneath every ``tool → ...`` header:

    - ``run_bash``     → shell command, bash-highlighted
    - ``run_python``   → Python source, python-highlighted
    - ``write_file``   → full file content, lexer guessed from path
    - ``edit_file``    → ``-``/``+`` diff block (old → new)
    - any other tool   → the one-line summary as a plain "text" block

    ``make_plan`` / ``send_response`` return empty summaries on purpose
    (see ``tool_summaries.py``) so we emit nothing for those — their
    payload is the stream that follows the header, not a static line.
    """
    match name:
        case ToolName.RUN_BASH:
            code = params.get("command", "")
            if code:
                return _code_block(code, "bash")
        case ToolName.RUN_PYTHON:
            code = params.get("code", "")
            if code:
                return _code_block(code, "python")
        case ToolName.WRITE_FILE:
            content = params.get("content", "")
            if content:
                return _code_block(content, _lexer_for_path(params.get("file_path", "")))
        case ToolName.EDIT_FILE:
            old = params.get("old_text", "")
            new = params.get("new_text", "")
            if old or new:
                lines = [f"- {line}" for line in old.splitlines()]
                if old and new:
                    lines.append("")
                lines.extend(f"+ {line}" for line in new.splitlines())
                return _code_block("\n".join(lines), "diff")
    # Fallback: render the full invocation as a python-style call so the
    # user sees the tool name AND all arguments together, not just an
    # isolated value like ``**/*.py``. ``reason`` is dropped because it's
    # already shown as the header label.
    args = {k: v for k, v in params.items() if k != "reason"}
    if args:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return _code_block(f"{name}({arg_str})", "python")
    return Text("")


def _tool_header_text(name: str, params: dict[str, Any]) -> Text:
    """Build the ``tool → {name}: {label}`` header as a two-tone ``Text``.

    Label text comes from the shared ``tool_header_label`` helper so
    this CLI stays in sync with the classic and rich variants. When
    no label is available (no reason, no summary) the trailing colon
    is omitted so the header reads as just ``tool → {name}``.

    Mirrors the ``user →`` / ``agent →`` body-to-label luminance
    relationship while keeping the orange hue unmistakably orange:

    - Tool body in ``#ffaf5f`` — vivid light orange (ANSI 215),
      saturation 100%, lightness ~69%, relative luminance ~186.
      Matches ``.user`` (Y~190) and ``.agent`` (Y~192) body brightness.
      Desaturated tans (``#d7af87``) at matching luminance read as
      beige rather than orange — full saturation here keeps the hue
      recognisable.
    - Tool prefix bold ``#ffaf00`` — yellow-leaning orange, full
      saturation, relative luminance ~179. Near-parity with the body
      (Δ ≈ −7), same body-to-label relationship as ``agent-label``
      (Δ ≈ −3) and a bright perceived weight.
    """
    t = Text()
    t.append("tool → ", style="bold #ffaf00")
    label = tool_header_label(name, params)
    body = f"{name}: {label}" if label else name
    t.append(body, style="#ffaf5f")
    return t


_HELP_TEXT = """Slash commands:
  /help   Show this help
  /clear  Clear the message history (UI only)
  /exit   Quit (also /quit)

Keys:
  Ctrl+L  Clear history
  Ctrl+C  Quit
  ↑ / ↓   Scroll history"""


# ───────────────────────────────────────────────────────────────────────────
# Widgets — one per output kind
# ───────────────────────────────────────────────────────────────────────────


class UserMessage(Static):
    """One-line ``user → {text}`` display.

    Mirrors the agent's label/text color pairing at matching brightness:
    the ``user →`` label is electric cyan (``#5fffff`` = color 87) just
    as ``agent-label`` is electric green (``#00ff5f`` = color 46), and
    the typed text is pastel cyan (``#5fd7d7``) paralleling ``.agent``'s
    pastel green (``#87d787``). Same "bright label over pastel body"
    relationship on both sides.
    """

    def __init__(self, text: str) -> None:
        content = Text()
        content.append("user → ", style="bold #5fffff")
        content.append(text, style="#5fd7d7")
        super().__init__(content, classes="user")


class ToolCall(Widget):
    """A tool call block — header, optional detail, optional running
    spinner, and a ✓/✗ result row once the tool completes.

    Owns its own lifecycle: ``start_running()`` mounts an inline
    spinner child; ``set_result()`` removes the spinner and mounts
    the status row. This keeps the running indicator tied to the
    specific tool it belongs to, which matters for parallel
    execution where multiple tools can be in-flight at once.

    Indented as a unit so all children share the same column.
    ``margin-top: 1`` guarantees one blank line above the block.
    ``background: transparent`` prevents Textual's themed background
    from painting a rectangle around the ``Syntax`` detail and
    mismatching the surrounding terminal color.
    """

    DEFAULT_CSS = """
    ToolCall {
        height: auto;
        padding-left: 2;
        margin-top: 1;
        background: black 0%;
    }
    ToolCall > Static {
        background: black 0%;
    }
    """

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        super().__init__()
        self._name = name
        self._params = params
        self._spinner: Spinner | None = None

    def compose(self) -> ComposeResult:
        yield Static(_tool_header_text(self._name, self._params), classes="tool-header")
        # Detail line is skipped when reason+summary would produce the same
        # text as the header label (otherwise we'd render it twice).
        if self._params.get("reason") and summarize_params(self._name, self._params):
            yield Static(
                _detail_renderable(self._name, self._params), classes="tool-detail"
            )

    def start_running(self) -> None:
        """Mount an inline spinner while the tool executes."""
        self._spinner = Spinner("running")
        self.mount(self._spinner)

    def set_result(self, is_error: bool, result: str) -> None:
        """Stop the spinner (if any) and mount the ✓/✗ status row."""
        if self._spinner is not None:
            self._spinner.remove()
            self._spinner = None
        if is_error:
            first_line = result.splitlines()[0] if result else ""
            self.mount(Static(f"error: {first_line} ✗", classes="tool-fail"))
        else:
            self.mount(Static("success ✓", classes="tool-ok"))


class MarkdownStream(Static):
    """Streaming markdown block for thinking / assistant text.

    Maintains an internal buffer and re-renders as ``Markdown`` on each
    delta. Textual batches redraws so the re-parse cost is bounded by
    the frame rate.

    Rich's ``Markdown`` doesn't inherit Textual CSS colors — it has its
    own internal style system. The per-kind base style is passed to
    ``Markdown(style=...)`` so body text picks up the intended color.
    """

    _BASE_STYLE: dict[str, str] = {
        "thinking": "dim italic #af87ff",
        "agent": "#87d787",
    }

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._buf = ""
        super().__init__("", classes=kind)

    def append_delta(self, delta: str) -> None:
        self._buf += delta
        self.update(Markdown(self._buf, style=self._BASE_STYLE[self._kind]))


class PromptArea(TextArea):
    """Multi-line prompt input that auto-expands up to ``max-height``.

    ``Enter`` submits the current buffer (posting ``Submitted``);
    ``Shift+Enter`` and ``Ctrl+J`` both insert a literal newline so
    users can compose multi-line prompts without committing. Empty
    submissions are swallowed by the handler upstream.

    Not every terminal distinguishes ``Shift+Enter`` from plain
    ``Enter`` — ``Ctrl+J`` is the universally supported alternative.
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter", "newline", "Newline", show=False),
        Binding("ctrl+j", "newline", "Newline", show=False),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter to send."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        value = self.text
        self.text = ""
        self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        self.insert("\n")


class Spinner(Static):
    """Dots spinner: ``{label} .`` → ``..`` → ``...`` cycling at 400ms."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._step = 1
        super().__init__(self._frame())

    def on_mount(self) -> None:
        self.set_interval(0.4, self._tick)

    def _tick(self) -> None:
        self._step = self._step % 3 + 1
        self.update(self._frame())

    def _frame(self) -> str:
        return f"{self._label} {'.' * self._step}"


class TokenCounter(Static):
    """Cumulative input/output token display, docked above the prompt.

    Reads its values from the ``Agent`` instance via ``update_totals``.
    Formatting is delegated to the module-level ``_format_token_line``
    helper so the counter and per-turn separator render identically.
    """

    def update_totals(
        self,
        delta_in: int, delta_cached: int, delta_out: int,
        cum_in: int, cum_cached: int, cum_out: int,
        context_tokens: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        cum_line = _format_token_line(
            cum_in, cum_cached, cum_out, label="cumulative",
            context_tokens=context_tokens,
            max_context_tokens=max_context_tokens,
        )
        if delta_in or delta_cached or delta_out:
            delta_line = _format_token_line(
                delta_in, delta_cached, delta_out, label="delta",
            )
            combined = Text()
            combined.append_text(delta_line)
            combined.append("\n")
            combined.append_text(cum_line)
            self.update(combined)
        else:
            self.update(cum_line)


# ───────────────────────────────────────────────────────────────────────────
# Permission modal
# ───────────────────────────────────────────────────────────────────────────


class InlinePermissionPrompt(Static):
    """Display-only "Allow execution? [Y/n]" row beneath a pending ToolCall.

    Replaces the former ``PermissionModal``. The tool widget directly
    above already shows the command/code being requested, so this row
    only needs to announce the prompt. The user answers by typing
    ``y`` (or empty) / ``n`` into the regular PromptArea and pressing
    Enter — ``AgentApp.on_prompt_submitted`` routes the input to the
    pending permission future instead of queueing a new prompt.

    Using the PromptArea for the answer (rather than widget-level
    bindings) sidesteps focus-management edge cases inside
    ``VerticalScroll`` and matches the classic CLI's answer flow.
    """

    def __init__(self) -> None:
        super().__init__("Allow execution? [Y/n]", classes="permission-prompt")


# ───────────────────────────────────────────────────────────────────────────
# Callback bridge
# ───────────────────────────────────────────────────────────────────────────


class TuiOutput:
    """Maps ``AgentCallbacks`` to ``AgentApp`` methods.

    All callbacks fire on the Textual event loop (same asyncio loop
    ``agent.process_input`` awaits on), so direct method calls are
    safe — no ``call_from_thread`` indirection. Every tool-related
    callback forwards its ``tool_use_id`` so the app can pair results
    to widgets precisely, even across parallel same-name calls.
    """

    def __init__(self, app: AgentApp) -> None:
        self._app = app

    def on_thinking(self, delta: str) -> None:
        """Route a make_plan thinking delta into the app's stream dispatcher."""
        self._app.stream_delta("thinking", delta)

    def on_text(self, delta: str) -> None:
        """Route a send_response text delta into the app's stream dispatcher."""
        self._app.stream_delta("agent", delta)

    def on_tool_start(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> None:
        """Announce a new auto-path tool call. Forwards to the app so the
        widget is registered in ``_pending`` keyed by ``tool_use_id``."""
        self._app.tool_start(tool_use_id, name, params)

    def on_tool_result(
        self,
        tool_use_id: str,
        name: str,
        params: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        """Deliver a tool's result to its pending widget.

        The app's ``tool_result`` looks up the widget by ``tool_use_id``
        — name-based pairing would be ambiguous when the model runs
        parallel calls to the same tool.
        """
        self._app.tool_result(tool_use_id, name, params, result, is_error)

    async def ask_permission(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> bool:
        """Passthrough to ``AgentApp.ask_permission_inline``.

        The app owns both the inline prompt widget lifecycle and the
        ``_pending`` registration, so this method only needs to satisfy
        the ``AgentCallbacks`` contract.
        """
        return await self._app.ask_permission_inline(tool_use_id, name, params)


# ───────────────────────────────────────────────────────────────────────────
# App
# ───────────────────────────────────────────────────────────────────────────


class AgentApp(App[None]):
    """Full-screen Textual TUI for the Python Agent."""

    CSS = _CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_history", "Clear"),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.output = TuiOutput(self)
        self.agent = Agent(
            client=make_client(settings),
            model=settings.model,
            tools=get_default_tools(),
            callbacks=AgentCallbacks(
                on_thinking=self.output.on_thinking,
                on_text=self.output.on_text,
                on_tool_start=self.output.on_tool_start,
                on_tool_result=self.output.on_tool_result,
                ask_permission=self.output.ask_permission,
            ),
            max_turns=settings.max_turns,
            max_context_tokens=fetch_max_context_tokens(settings),
            prune_context_threshold=settings.prune_context_threshold,
        )
        self._active_stream: MarkdownStream | None = None
        self._active_stream_kind: str | None = None
        # Pre-stream / pre-API "thinking…" spinner, used only between
        # user submit and the first streaming delta. Not per-tool.
        self._active_spinner: Spinner | None = None
        # Pending tool calls keyed by ``tool_use_id``. Entries are added
        # when ``on_tool_start`` fires (auto-path) or when
        # ``ask_permission_inline`` returns (approve-path — registered
        # whether approved or denied, since the processor emits a
        # tool_result for denials too). Popped when the matching
        # ``on_tool_result`` arrives. Parallel same-name calls stay
        # correctly paired because IDs are unique per call.
        self._pending: dict[str, ToolCall] = {}
        # Prompt queue: submissions land here; a single background worker
        # (_drain_prompts) runs them sequentially. Lets the user type
        # ahead while a turn is still executing.
        self._prompt_queue: asyncio.Queue[str] = asyncio.Queue()
        # Set by ``ask_permission_inline`` while waiting for a y/n
        # answer typed into the PromptArea. ``on_prompt_submitted``
        # detects a pending future and resolves it instead of queueing
        # the input as a new user prompt.
        self._pending_permission: asyncio.Future[bool] | None = None
        self._turn_count: int = 0

    def compose(self) -> ComposeResult:
        """Build the static layout: scrollable message pane + token counter + input box."""
        yield VerticalScroll(id="messages")
        yield TokenCounter(id="token-counter")
        yield PromptArea(id="prompt")

    def on_mount(self) -> None:
        """Render the startup banner + hints and focus the prompt."""
        self._mount_to_messages(Static(_BANNER, classes="banner"))
        self._mount_blank()
        self._mount_hint(f"model:     {self.settings.model}")
        self._mount_hint(f"provider:  {self.settings.api_provider}")
        self._mount_hint("help:      /help for commands · Enter send · Shift+Enter newline · Ctrl+C quit")
        # Initialize the cumulative counter so it reads a styled zero
        # line from startup instead of being blank until the first turn.
        self.query_one(TokenCounter).update_totals(0, 0, 0, 0, 0, 0)
        self.query_one("#prompt", PromptArea).focus()
        # Start the single long-running consumer that drains the prompt
        # queue. Runs for the whole app lifetime.
        self._drain_prompts()

    # ── callback targets ───────────────────────────────────────────────

    def stream_delta(self, kind: str, delta: str) -> None:
        """Append a streaming delta to the active block, or open a new one.

        On kind-change: closes any active pre-stream spinner, emits a
        visual blank line, mounts a fresh ``MarkdownStream`` widget and
        (for the agent-response stream) an ``agent →`` label above it.
        Subsequent deltas of the same kind append to the existing widget.

        Always scrolls the messages pane to the bottom so streaming
        content stays visible — ``Static.update`` re-render alone does
        not trigger ``VerticalScroll`` auto-scroll.
        """
        if self._active_stream_kind != kind:
            self._close_spinner()
            self._mount_blank()
            if kind == "agent":
                self._mount_to_messages(Static("agent →", classes="agent-label"))
                self._mount_blank()
            self._active_stream = MarkdownStream(kind)
            self._active_stream_kind = kind
            self._mount_to_messages(self._active_stream)
        assert self._active_stream is not None
        self._active_stream.append_delta(delta)
        # ``update()`` doesn't remount, so _mount_to_messages' scroll_end
        # doesn't fire for subsequent deltas. Explicitly scroll here so
        # streaming content stays visible at the bottom.
        self.query_one("#messages", VerticalScroll).scroll_end(animate=False)

    def tool_start(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> None:
        """Auto-path tool kickoff. Mount the widget and start its spinner.

        Intercepted tools (``make_plan`` / ``send_response``) skip the
        spinner and the pending-dict entry because they have no
        ``on_tool_result`` — their lifecycle is carried by the stream
        that follows.
        """
        self._close_active_stream_or_spinner()
        tool = ToolCall(name, params)
        self._mount_to_messages(tool)
        if name not in INTERCEPTED_TOOLS:
            tool.start_running()
            self._pending[tool_use_id] = tool

    def tool_result(
        self,
        tool_use_id: str,
        name: str,
        params: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        """Close the matching pending tool widget with its result.

        Every tool that produces a result must have been registered in
        ``_pending`` earlier — auto-path via ``tool_start`` or
        approve-path via ``ask_permission_inline``. The assertion
        encodes that invariant; a missing entry would mean we took a
        code path that doesn't exist yet and is a real bug, not
        something to paper over.
        """
        del name, params  # widget for this ID already carries them
        self._close_active_stream_or_spinner()
        tool = self._pending.pop(tool_use_id, None)
        assert tool is not None, (
            f"tool_result for unregistered tool_use_id {tool_use_id!r}"
        )
        tool.set_result(is_error, result)
        # set_result mounts a child inside ToolCall — VerticalScroll
        # doesn't auto-scroll for descendant mounts, so trigger it.
        self.query_one("#messages", VerticalScroll).scroll_end(animate=False)

    async def ask_permission_inline(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> bool:
        """Mount the pending ToolCall + inline permission prompt, await decision.

        Replaces the former ``PermissionModal`` flow. The tool widget
        renders the header + params; a single-line
        ``InlinePermissionPrompt`` announces the decision. The user
        answers by typing ``y`` / ``n`` (or just Enter for yes) into
        the PromptArea — ``on_prompt_submitted`` resolves the future
        stored on ``self._pending_permission``.

        The tool widget is registered in ``_pending`` **regardless** of
        the decision: the processor fires ``on_tool_result`` for denied
        tools too (processor.py:359 with ``"Permission denied"``), and
        ``tool_result`` asserts the widget is present. On approval, the
        spinner starts; on denial, ``ToolCall.set_result`` guards its
        spinner removal so the denial renders cleanly as an error row.
        """
        # Match the auto-path pattern in tool_start — close any pre-tool
        # spinner or stream so nothing lingers beneath the prompt.
        self._close_active_stream_or_spinner()

        tool = ToolCall(name, params)
        self._mount_to_messages(tool)
        prompt = InlinePermissionPrompt()
        self._mount_to_messages(prompt)

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_permission = future
        try:
            approved = await future
        finally:
            self._pending_permission = None
            prompt.remove()

        self._pending[tool_use_id] = tool
        if approved:
            tool.start_running()
        return approved

    # ── lifecycle helpers ──────────────────────────────────────────────

    def _mount_to_messages(self, widget: Widget) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.mount(widget)
        messages.scroll_end(animate=False)

    def _mount_blank(self) -> None:
        """Mount an empty spacer row for vertical breathing room."""
        self._mount_to_messages(Static("", classes="spacer"))

    def _mount_spinner(self, label: str) -> None:
        self._close_spinner()
        self._active_spinner = Spinner(label)
        self._mount_to_messages(self._active_spinner)

    def _close_spinner(self) -> None:
        if self._active_spinner is not None:
            self._active_spinner.remove()
            self._active_spinner = None

    def _forget_active_stream(self) -> None:
        """Drop the reference to the current streaming widget.

        The ``MarkdownStream`` widget stays mounted in the scroll area
        (the rendered thinking / response text remains visible); this
        just clears our handles so the next delta kind-change mounts a
        fresh widget instead of appending to the closed one.
        """
        self._active_stream = None
        self._active_stream_kind = None

    def _close_active_stream_or_spinner(self) -> None:
        """Close the pre-stream spinner and any active streaming block.

        Pending tool widgets are owned by the ``_pending`` dict and
        keyed by ``tool_use_id`` — they close themselves when their
        matching ``on_tool_result`` fires and should *not* be touched
        here (doing so would drop spinner state for in-flight tools).
        """
        self._close_spinner()
        self._forget_active_stream()

    def _mount_user_message(self, text: str) -> None:
        self._mount_to_messages(UserMessage(text))

    def _mount_hint(self, text: str) -> None:
        self._mount_to_messages(Static(text, classes="hint"))

    # ── event handlers ─────────────────────────────────────────────────

    @on(PromptArea.Submitted)
    async def on_prompt_submitted(self, event: PromptArea.Submitted) -> None:
        """Route submissions: permission answer, slash command, or queued prompt.

        When ``ask_permission_inline`` is awaiting a decision, the next
        submission is treated as the answer (empty / ``y`` / ``yes`` →
        approve, anything else → deny), matching classic-CLI behaviour.

        Otherwise, prompts are enqueued for the consumer to run.
        Nothing else is mounted here: the spinner/stream tracker is
        single-slot (see ``_mount_spinner``), so mounting during an
        active stream would corrupt it. The user message also mounts
        in the consumer to preserve execution order — queued prompts
        appearing above the active turn's content would be misleading.

        Feedback that the submission was accepted comes from
        ``PromptArea.action_submit`` clearing the input box.
        """
        value = event.value.strip()
        if self._pending_permission is not None and not self._pending_permission.done():
            approved = value.lower() in ("", "y", "yes")
            self._pending_permission.set_result(approved)
            return
        if not value:
            return
        if value.startswith("/"):
            await self._handle_slash(value)
            return
        self._prompt_queue.put_nowait(value)

    @work(exclusive=True, name="prompt-drain")
    async def _drain_prompts(self) -> None:
        """Single long-running worker. Pulls prompts and runs turns sequentially.

        Runs inside a worker context so ``ask_permission`` (called via
        ``_run_one_turn`` → ``agent.process_input``) can
        ``await push_screen_wait(...)`` — Textual 8.x requires modal
        value-returning awaits to originate from a worker.

        The outer try/except is belt-and-suspenders: ``_run_one_turn``
        already catches turn-level exceptions, but an unexpected raise
        outside that scope would otherwise silently kill the consumer
        and stop all queueing.
        """
        while True:
            value = await self._prompt_queue.get()
            try:
                self._mount_user_message(value)
                self._mount_spinner("thinking")
                await self._run_one_turn(value)
            except Exception:
                logger.exception("drain loop: turn raised unexpectedly")
                self._close_active_stream_or_spinner()

    async def _run_one_turn(self, value: str) -> None:
        """Run one agent turn.

        After the turn ends (success or exception), mounts a horizontal
        rule summarizing that turn's token delta, and refreshes the
        cumulative counter docked above the prompt. Deltas are derived
        by snapshot-subtract so no per-turn state is added to the Agent.
        """
        before_in = self.agent.total_input_tokens
        before_cached = self.agent.total_cached_input_tokens
        before_out = self.agent.total_output_tokens
        try:
            await self.agent.process_input(value)
        except Exception as e:
            self._mount_hint(f"Error: {e}")
        finally:
            self._close_active_stream_or_spinner()
            turn_in = self.agent.total_input_tokens - before_in
            turn_cached = self.agent.total_cached_input_tokens - before_cached
            turn_out = self.agent.total_output_tokens - before_out
            self._turn_count += 1
            self._mount_blank()
            self._mount_blank()
            self._mount_to_messages(Static(
                Rule(
                    title=f"interaction {self._turn_count}",
                    align="left",
                    style="color(245)",
                    characters="-",
                ),
                classes="turn-separator",
            ))
            self.query_one(TokenCounter).update_totals(
                turn_in, turn_cached, turn_out,
                self.agent.total_input_tokens,
                self.agent.total_cached_input_tokens,
                self.agent.total_output_tokens,
                self.agent.last_context_tokens,
                self.agent.max_context_tokens,
            )

    async def _handle_slash(self, cmd: str) -> None:
        # Single dispatch point for every slash command.
        match cmd:
            case "/help":
                self._mount_hint(_HELP_TEXT)
            case "/clear":
                self.action_clear_history()
            case "/exit" | "/quit":
                self.exit()
            case _:
                self._mount_hint(f"unknown command: {cmd}")

    def action_clear_history(self) -> None:
        """Ctrl+L / /clear — empty the UI history (agent.messages is untouched)."""
        # Drain any queued prompts so they don't appear after the clear.
        while not self._prompt_queue.empty():
            self._prompt_queue.get_nowait()
        # Resolve any pending permission as denied — otherwise the
        # consumer would block forever on a future whose prompt widget
        # we're about to remove.
        if self._pending_permission is not None and not self._pending_permission.done():
            self._pending_permission.set_result(False)
        self.query_one("#messages", VerticalScroll).remove_children()
        self._close_active_stream_or_spinner()
        self._pending.clear()
        self._turn_count = 0
        self.query_one(TokenCounter).update_totals(0, 0, 0, 0, 0, 0)


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────


def run(settings: Settings) -> None:
    """Launch the Textual TUI — called from ``__main__.py``."""
    logging_config.configure(log_dir=settings.log_dir)
    AgentApp(settings).run()
