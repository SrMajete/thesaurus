"""Textual-based CLI — full-screen TUI variant.

Parallel to ``cli.py`` (classic) and ``cli_rich.py`` (Rich scroll-based
REPL). This one uses Textual for a true TUI experience:

- fixed ``Input`` docked at the bottom (never scrolls away)
- scrollable ``VerticalScroll`` message history above
- mouse support, Ctrl+L clear, Ctrl+C/D quit
- collapsible tool-call blocks with inline spinners and result rows
- live-streaming markdown for thinking and assistant response
- modal permission prompts

Implements the same ``AgentCallbacks`` contract as the other two CLIs,
so the core (``agent.py``, ``processor.py``, ``api_client.py``, tools)
is unchanged. Callbacks fire on the Textual event loop (same asyncio
loop ``agent.process_input`` awaits on), so no thread-bridge indirection
is needed.
"""

from __future__ import annotations

from typing import Any

from pyfiglet import Figlet
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Static, TextArea

from . import logging_config
from .agent import Agent, AgentCallbacks
from .client import make_client
from .config import Settings
from .tool_summaries import INTERCEPTED_TOOLS, summarize_params, tool_header_label
from .tools import get_default_tools
from .tools.base import ToolName


# ───────────────────────────────────────────────────────────────────────────
# Theme — single Textual CSS string
# ───────────────────────────────────────────────────────────────────────────

# Palette mirrors ``cli_rich.py`` (and classic ``cli.py``) — semantic
# class names bind to the same ANSI-256 shades used elsewhere so all
# three CLIs read as one product. Hex values below are the standard
# xterm 256-color mappings for ANSI color indices:
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

#prompt {
    dock: bottom;
    border: round #87afff;
    height: auto;
    min-height: 3;
    max-height: 8;
    margin: 0;
    padding: 0 1;
}

#prompt:focus {
    border: round #00ff5f;
}

.banner {
    color: #00aaaa;
    text-style: bold;
    padding: 1 0 0 1;
    height: auto;
}

.user {
    padding: 1 0 0 0;
}

.hint {
    color: #8a8a8a;
    text-style: italic;
}

.tool-header {
    color: #ff8700;
    text-style: bold;
}

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
   reads as content belonging to the enclosing ``make_plan`` block. */
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

PermissionModal {
    align: right middle;
}

/* Half-screen right pane: 50% width, full height. The detail row
   takes 1fr so long commands / code blocks fill the vertical space;
   buttons dock at the bottom. */
#permission-dialog {
    grid-size: 2;
    grid-gutter: 1 2;
    grid-rows: auto 1fr 3;
    padding: 1 2;
    width: 50%;
    height: 100%;
    border: solid #00aaaa;
    background: $surface;
}

#permission-dialog > .dialog-header {
    column-span: 2;
    color: #ff8700;
    text-style: bold;
}

#permission-dialog > .dialog-detail {
    column-span: 2;
    /* No color override — see .tool-detail comment above. */
}

#permission-dialog Button {
    width: 100%;
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


def _tool_header_text(name: str, params: dict[str, Any]) -> str:
    """Build the ``tool → {name}: {label}`` header string.

    Label text comes from the shared ``tool_header_label`` helper so
    this CLI stays in sync with the classic and rich variants. When
    no label is available (no reason, no summary) the trailing colon
    is omitted so the header reads as just ``tool → {name}``.
    """
    label = tool_header_label(name, params)
    return f"tool → {name}: {label}" if label else f"tool → {name}"


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

    Label and text carry different styles so the prompt marker reads as
    the structural element (bold blue) and the typed content stands out
    in a lighter shade — same relationship ``cli_rich`` uses.
    """

    def __init__(self, text: str) -> None:
        content = Text()
        content.append("user → ", style="bold #0000ff")
        content.append(text, style="#87afff")
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


# ───────────────────────────────────────────────────────────────────────────
# Permission modal
# ───────────────────────────────────────────────────────────────────────────


class PermissionModal(ModalScreen[bool]):
    """Blocking modal: approve or deny a tool call. Returns ``bool``."""

    BINDINGS = [
        Binding("escape", "deny", show=False),
        Binding("y", "allow", show=False),
        Binding("n", "deny", show=False),
    ]

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        super().__init__()
        self._name = name
        self._params = params

    def compose(self) -> ComposeResult:
        yield Grid(
            Static(
                _tool_header_text(self._name, self._params), classes="dialog-header"
            ),
            Static(
                _detail_renderable(self._name, self._params), classes="dialog-detail"
            ),
            Button("Allow (Y)", variant="success", id="ok"),
            Button("Deny (N)", variant="error", id="deny"),
            id="permission-dialog",
        )

    def on_mount(self) -> None:
        # Focus Allow so Enter approves without needing to click/Tab first.
        self.query_one("#ok", Button).focus()

    @on(Button.Pressed, "#ok")
    def _allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def _deny(self) -> None:
        self.dismiss(False)

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


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
        """Open the permission modal and block until the user dismisses it.

        On approval, registers the tool widget with a spinner via
        ``register_approved_tool`` so the user sees all approved tools
        appear in approval order while they run in parallel.

        ``push_screen_wait`` requires a worker context (Textual 8.x).
        ``agent.process_input`` runs as a ``@work``-decorated method on
        the App, so direct ``await`` here is correct.
        """
        approved = bool(
            await self._app.push_screen_wait(PermissionModal(name, params))
        )
        if approved:
            self._app.register_approved_tool(tool_use_id, name, params)
        return approved


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
        )
        self._active_stream: MarkdownStream | None = None
        self._active_stream_kind: str | None = None
        # Pre-stream / pre-API "thinking…" spinner, used only between
        # user submit and the first streaming delta. Not per-tool.
        self._active_spinner: Spinner | None = None
        # Pending tool calls keyed by ``tool_use_id``. Entries are added
        # when ``on_tool_start`` fires (auto-path) or ``ask_permission``
        # returns True (approve-path); popped when the matching
        # ``on_tool_result`` arrives. Parallel same-name calls stay
        # correctly paired because IDs are unique per call.
        self._pending: dict[str, ToolCall] = {}

    def compose(self) -> ComposeResult:
        """Build the static layout: scrollable message pane + input box."""
        yield VerticalScroll(id="messages")
        yield PromptArea(id="prompt")

    def on_mount(self) -> None:
        """Render the startup banner + hints and focus the prompt."""
        self._mount_to_messages(Static(_BANNER, classes="banner"))
        self._mount_blank()
        self._mount_hint(f"model:     {self.settings.model}")
        self._mount_hint(f"provider:  {self.settings.api_provider}")
        self._mount_hint("help:      /help for commands · Enter send · Shift+Enter newline · Ctrl+C quit")
        self.query_one("#prompt", PromptArea).focus()

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
        approve-path via ``register_approved_tool``. The assertion
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

    def register_approved_tool(
        self, tool_use_id: str, name: str, params: dict[str, Any]
    ) -> None:
        """Mount a tool widget for an approve-path tool just approved.

        Called from ``TuiOutput.ask_permission`` after the modal
        returns True. The processor skips ``on_tool_start`` for
        permission-required tools, so this is our only chance to show
        the inline block before the tool runs.
        """
        tool = ToolCall(name, params)
        self._mount_to_messages(tool)
        tool.start_running()
        self._pending[tool_use_id] = tool

    # ── lifecycle helpers ──────────────────────────────────────────────

    def _mount_to_messages(self, widget: Widget) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.mount(widget)
        messages.scroll_end(animate=False)

    def _mount_blank(self) -> None:
        """Mount an empty spacer row — the TUI analogue of cli_rich's ``_blank``."""
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
        value = event.value.strip()
        if not value:
            return
        if value.startswith("/"):
            await self._handle_slash(value)
            return
        self._mount_user_message(value)
        self._mount_spinner("thinking")
        self._run_turn(value)

    @work(exclusive=True)
    async def _run_turn(self, value: str) -> None:
        """Run one agent turn as a Textual worker.

        Must run inside a worker context so ``ask_permission`` can
        ``await push_screen_wait(...)`` — Textual 8.x requires modal
        value-returning awaits to originate from a worker.
        """
        try:
            await self.agent.process_input(value)
        except Exception as e:
            self._mount_hint(f"Error: {e}")
        finally:
            self._close_active_stream_or_spinner()

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
        self.query_one("#messages", VerticalScroll).remove_children()
        self._close_active_stream_or_spinner()
        self._pending.clear()


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────


def run(settings: Settings) -> None:
    """Launch the Textual TUI — called from ``__main__.py``."""
    logging_config.configure(debug=settings.debug, log_dir=settings.log_dir)
    AgentApp(settings).run()
