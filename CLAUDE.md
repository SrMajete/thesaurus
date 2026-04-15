# Python Agent

A streaming AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns.

## Architecture

```
cli.py          ┐
cli_rich.py     ┤→ agent.py (state) → processor.py (logic) → api_client.py (API)
cli_tui.py      │                            ↕
(I/O frontends) │                       tools/*.py (execution)
                │
shared: client.py (API client factory) · tool_summaries.py (SUMMARIZERS, INTERCEPTED_TOOLS)
```

- **Agent** owns state (messages, plan, tools, callbacks). One instance per session.
- **Processor** owns the loop logic. Pure — no I/O, no state.
- **API client** streams with jiter partial JSON parsing (`make_plan.thinking`, `make_plan.roadmap`, `send_response.response`).
- **Tools** satisfy a `Tool` Protocol via class-level attributes. No inheritance. Adding one: new class + `ToolName` entry + registry line + summarizer entry.
- **Callbacks** (`AgentCallbacks` dataclass) decouple the agent from I/O. Every tool-related callback carries a `tool_use_id` so consumers can pair start/result — `name` collides on parallel calls. TUI keys on the ID; classic/rich pair positionally.

## Good Coding Principles

1. **KISS** — Simplest thing that works. Clever is a liability.
2. **DRY** — One authoritative source per fact; collapse duplication on the third occurrence, not the second.
3. **YAGNI** — Build for today's requirement, not tomorrow's guess.
4. **SRP** (Single Responsibility) — One reason to change per unit; expressible in one sentence without "and".
5. **OCP** (Open/Closed) — Extend behavior by adding new code, not by editing working code.
6. **LSP** (Liskov Substitution) — A subtype must be usable anywhere its base type is, with no surprises.
7. **ISP** (Interface Segregation) — Callers depend only on the methods they actually use; split fat interfaces.
8. **DIP** (Dependency Inversion) — Depend on interfaces; inject concrete types at the edges.
9. **Separation of Concerns** — Each module owns one concern; don't mix transport, logic, and storage.
10. **Law of Demeter** — Talk only to immediate collaborators; `a.b.c.d()` is a smell.
11. **Composition over Inheritance** — Assemble behavior from small pieces; inheritance is for substitutability, not reuse.
12. **Principle of Least Astonishment** — If the name surprises the reader, rename it or change the behavior.
13. **Fail Fast** — Validate once at the boundary; trust invariants inside.
14. **Boy Scout Rule** — Leave touched code cleaner than you found it, within the scope of your change.
15. **Encapsulation** — Hide internals behind stable interfaces; expose intent, not structure.
16. **High Cohesion / Loose Coupling** — Related code together, unrelated code apart.

## Self-Review Checklist (MANDATORY)

Before presenting a plan or accepting work as done:

1. **Correct** — solves the stated problem, no silent regressions.
2. **Simple** — simplest thing that works; no abstractions without a load-bearing reason.
3. **Elegant** — reads as if always meant to be this way.
4. **Well-written** — proper typing, idiomatic Python, rich docstrings where intent isn't obvious.
5. **Maintainable** — one source of truth per concept.
6. **No unused / irrelevant / over-complicated code** — no dead imports, no speculative flexibility. Unused parameters required by callback contracts are deleted with `del`.
7. **Principle-compliant** — name which principles the change upholds *and* which trade-offs are deliberate. Silent trade-offs are bugs.

## Key Design Decisions

- `tool_choice: any` — every API response must use a tool. No freeform text.
- `make_plan` + `send_response` are intercepted (`is_intercepted = True`). `execute()` is never called; the processor persists the plan and streams the response. `INTERCEPTED_TOOLS` is derived from the flag.
- `agent.plan` is injected into the system prompt every turn via `_current_plan_section()`, so the model evolves its own roadmap across turns.
- Two-phase tool dispatch: `_collect_decisions` (batch permissions upfront) → `_execute_decisions`. `needs_permission` × `is_parallelizable` gives four dispatch quadrants.
- `reason` auto-injected as the first property of every tool schema so the model writes intent before payload, enabling streaming headers.
- Prompt caching: static system cached, dynamic plan uncached, tools cached via last-tool `cache_control`, last user message gets an ephemeral breakpoint.
- `run_loop` wraps the body after the assistant-message append in `try/finally`; any exception (KeyboardInterrupt, callback failure) that would orphan `tool_use` blocks falls through to stub `tool_result`s, preserving the API's block-pairing invariant.

## Conventions

- **Python 3.12.10**, pyenv virtualenv `python-agent`
- **Tool names:** `verb_noun` snake_case, regex-checked at `Agent.__init__`
- **`ToolName` StrEnum:** single source of truth for tool names
- **No code in `__init__.py`** — re-exports only

## Provider & CLI

`API_PROVIDER` in `.env`: `anthropic` (default, needs `ANTHROPIC_API_KEY`) or `bedrock` (AWS creds + ARN inference-profile model IDs).

```bash
python -m python_agent                     # rich UI (default)
CLI_STYLE=classic python -m python_agent   # hand-rolled ANSI, no third-party TUI deps
CLI_STYLE=tui     python -m python_agent   # full-screen Textual TUI (requires real TTY)
```

## Adding a Tool

1. Create `python_agent/tools/your_tool.py` with a class satisfying the `Tool` Protocol.
2. Add `YOUR_TOOL = "your_tool"` to `ToolName` in `tools/base.py`.
3. Register in `tools/registry.py`.
4. Add a summarizer entry in `tool_summaries.py` `SUMMARIZERS`.
5. Update WORKFLOW rule 2 in `prompts.py` if the tool replaces a `run_bash` pattern.

## Built-in Tools

- **File I/O:** `read_file`, `write_file`, `edit_file`
- **Search:** `glob_files`, `grep_files`
- **Execution:** `run_bash`, `run_python`
- **Agent control:** `make_plan`, `send_response` (intercepted)
