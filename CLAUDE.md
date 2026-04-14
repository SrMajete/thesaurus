# Python Agent

A streaming AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns — modular, expandable, easy to add tools and techniques.

## Architecture

```
cli.py          ┐
cli_rich.py     ┤→ agent.py (state) → processor.py (logic) → api_client.py (API)
(I/O frontends) │                            ↕
                │                       tools/*.py (execution)
                │
shared: client.py (API client factory) · tool_summaries.py (tool → preview)
```

- **Agent** owns state (messages, plan, tools, callbacks). One instance per session.
- **Processor** owns the loop logic. Pure — no I/O, no state. Reads/writes the Agent instance.
- **API client** handles streaming with jiter partial JSON parsing. Streams `make_plan.thinking`, `make_plan.roadmap`, and `send_response.response` live.
- **Tools** satisfy a `Tool` Protocol with class-level attributes. No inheritance. Adding a tool: one class, one `ToolName` enum entry, one registry line, one summarizer entry.
- **Callbacks** (`AgentCallbacks` dataclass) decouple the agent from terminal I/O.
- **Two CLI frontends** share everything except rendering: `cli.py` (hand-rolled ANSI, classic fallback) and `cli_rich.py` (Rich library — themed, markdown rendering, spinner). One `cli_style` setting picks which runs. See **CLI Variants** below.

## Quality Standards

All code must follow good coding and architecture principles: KISS, DRY, YAGNI, SRP, OCP, LSP, ISP, DIP, Separation of Concerns, Law of Demeter, Composition over Inheritance, Principle of Least Astonishment, Fail Fast, Boy Scout Rule, Encapsulation, and High Cohesion / Loose Coupling. See `prompts.py` CODE_QUALITY section for the full definitions. These apply to every change — code, finding, recommendation, or plan.

## Self-Review Checklist (MANDATORY)

**On every planning change and on every code review**, before presenting a plan or accepting your own work as done, verify the solution against this checklist:

1. **Correct** — does it actually solve the stated problem, with verified assumptions and no silent regressions?
2. **Simple** — is this the simplest thing that works? No state machines or abstractions without a load-bearing reason.
3. **Elegant** — does the design read as if it were always meant to be this way, not like something bolted on?
4. **Well-written** — proper typing, idiomatic Python, clear naming, rich docstrings where intent isn't obvious from the code.
5. **Maintainable** — a single source of truth per concept; changing one behavior means editing one place, not N.
6. **No unused / irrelevant / over-complicated code** — no dead imports, no unused parameters (except ones required by a callback contract, explicitly deleted with `del`), no speculative flexibility, no premature abstraction.
7. **Complies with all good coding/architecture principles** — explicitly name which principles the change upholds *and* which trade-offs you are deliberately making (e.g. "duplicates X for KISS; a shared module wouldn't earn its weight for 4 lines").

If any of these fails, fix it or **name it as a deliberate trade-off with reasoning** before moving on. Quality is non-negotiable — *unless* you explicitly acknowledge you're choosing otherwise and why. Silent trade-offs are bugs.

## Key Design Decisions

- `tool_choice: any` — every API response must use a tool. No freeform text.
- `make_plan` + `send_response` are intercepted tools — their `execute()` is never called. The processor handles them specially (persist plan, stream response).
- `agent.plan` is injected into the system prompt every turn via `_current_plan_section()`. The model sees its own thinking + roadmap and evolves it.
- Two-phase tool dispatch: `_collect_decisions` (batch permissions upfront) → `_execute_decisions` (parallel or serial execution).
- `needs_permission` × `is_parallelizable` flags produce four dispatch quadrants.
- `reason` field auto-injected as first property of every tool schema so the model writes intent before payload, enabling streaming tool headers.
- Prompt caching: static system sections in one cached block, dynamic plan uncached, tool definitions cached via last-tool `cache_control`, last user message gets ephemeral breakpoint.
- Client construction lives in `client.py` (`make_client(settings)`) so both CLIs share one factory.
- Tool display summaries live in `tool_summaries.py` (`SUMMARIZERS` dict, typed with `ToolName` keys) so both CLIs render identical headers.

## CLI Variants

The default frontend is `cli_rich.py`. Override with `CLI_STYLE=classic` in `.env` to use the original `cli.py`. Dispatch happens once at startup in `__main__.py`:

- **rich** — `Theme` for styling, `Live` + `_Dots` dots-text spinner (`thinking .` / `thinking ..` / `thinking ...`), `Markdown` swap on stream end for syntax-highlighted code blocks / lists / emphasis, indented layout under the `agent →` label, `Panel` header.
- **classic** — hand-rolled ANSI constants, `_DisplayState` transition machine, `_print_wrapped` word-wrap, ✓/✗ in-place result completion. Stable, no third-party TUI dependency.

Both implement the same `AgentCallbacks` contract; everything above the I/O layer is identical.

## Conventions

- **Python 3.12.10**, pyenv virtualenv `python-agent`
- **Tool names:** `verb_noun` snake_case, enforced by regex in `Agent.__init__`
- **`ToolName` StrEnum:** single source of truth for tool names across all consumers
- **No code in `__init__.py`** — re-exports only
- **Rich docstrings** on all modules, classes, and functions
- **Simple but top-notch:** elegant architecture, proper typing, idiomatic Python, clean abstractions
- **KISS philosophy:** simplest thing that works, clever is a liability

## Provider Support

Set `API_PROVIDER` in `.env`:

- `anthropic` (default) — requires `ANTHROPIC_API_KEY`
- `bedrock` — uses AWS credentials (`AWS_PROFILE`, `AWS_REGION`). Model IDs are ARN-based inference profiles (e.g., `arn:aws:bedrock:eu-central-1:...:inference-profile/eu.anthropic.claude-sonnet-4-6`)

Set `CLI_STYLE` in `.env`:

- `rich` (default) — Rich-library UI (`cli_rich.py`)
- `classic` — hand-rolled ANSI UI (`cli.py`)

## Running

```bash
cd python-agent
python -m python_agent          # rich UI (default)
CLI_STYLE=classic python -m python_agent   # classic UI
```

## Adding a Tool

1. Create `python_agent/tools/your_tool.py` with a class satisfying the `Tool` Protocol
2. Add `YOUR_TOOL = "your_tool"` to `ToolName` in `tools/base.py`
3. Register in `tools/registry.py`
4. Add a summarizer in `tool_summaries.py` `SUMMARIZERS` dict (keyed by `ToolName` member)
5. Update WORKFLOW rule 2 in `prompts.py` if the tool replaces a `run_bash` pattern

## Built-in Tools

- **File I/O:** `read_file`, `write_file`, `edit_file`
- **Search:** `glob_files`, `grep_files`
- **Execution:** `run_bash`, `run_python`
- **Agent control:** `make_plan`, `send_response` (both intercepted; their `execute()` is never called)

## Reference Code

The original TypeScript Claude Code CLI lives at `../original_code/`. Use it to check how the original implements a feature before building the Python equivalent — but write fresh idiomatic Python, don't port the TypeScript.

## Known Issues

- No context management — long sessions will hit the context window limit
- KeyboardInterrupt during tool execution leaves orphaned `tool_use` blocks (corrupts conversation state)
- Terminal output can interleave during parallel auto-approved tools (cosmetic, documented in `processor.py`)
- Classic `cli.py` `_fit` measures `len(text)` not terminal columns (emoji/CJK off by one). Rich CLI uses Rich's native display-width and is unaffected.
