# Python Agent

A streaming AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns.

## Prime directive

Every change in this codebase — new code, reviews, refactors, bug fixes, style edits — must pass the **Self-Review Checklist** below and comply with the **Good Coding Principles**. This applies equally to one-line fixes and major refactors. If a proposed change fails the checklist, drop it.

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

Run at three points: before presenting a plan, before writing code, and before declaring work done. Not optional.

1. **Correct** — solves the stated problem, no silent regressions.
2. **Simple** — simplest thing that works; no abstractions without a load-bearing reason.
3. **Elegant** — reads as if always meant to be this way.
4. **Well-written** — proper typing, idiomatic Python, rich docstrings where intent isn't obvious.
5. **Maintainable** — one source of truth per concept.
6. **No unused / irrelevant / over-complicated code** — no dead imports, no speculative flexibility. Unused parameters required by callback contracts are deleted with `del`.
7. **Principle-compliant** — name which principles the change upholds *and* which trade-offs are deliberate. Silent trade-offs are bugs.

## Architecture

```
cli_tui.py → agent.py (state) → processor.py (logic) → api_client.py (API)
                                        ↕
                                   tools/*.py (execution)

shared helpers: client.py (API client factory) · tool_summaries.py (SUMMARIZERS, INTERCEPTED_TOOLS)
```

- **Agent** owns state (messages, plan, tools, callbacks). One instance per session.
- **Processor** owns the loop logic. Pure — no I/O, no state.
- **API client** streams with jiter partial JSON parsing (`make_plan.thinking`, `make_plan.roadmap`, `send_response.response`).
- **Tools** satisfy a `Tool` Protocol via class-level attributes. No inheritance.
- **Callbacks** (`AgentCallbacks` dataclass) decouple the agent from I/O. Every tool-related callback carries a `tool_use_id` so the TUI can pair start/result precisely — `name` collides on parallel calls.

## Key Design Decisions

- `tool_choice: any` — every API response must use a tool. No freeform text.
- `make_plan` + `send_response` are intercepted (`is_intercepted = True`). `execute()` is never called; the processor persists the plan and streams the response. `INTERCEPTED_TOOLS` is derived from the flag.
- `agent.plan` is injected into the system prompt every turn via `_current_plan_section()`, so the model evolves its own roadmap across turns.
- Two-phase tool dispatch: `_collect_decisions` (batch permissions upfront) → `_execute_decisions`. `needs_permission` × `is_parallelizable` gives four dispatch quadrants.
- `reason` auto-injected as the first property of every tool schema so the model writes intent before payload, enabling streaming headers.
- Prompt caching: static system cached, dynamic plan uncached, tools cached via last-tool `cache_control`, last user message gets an ephemeral breakpoint.
- `run_loop` wraps the body after the assistant-message append in `try/finally`; any exception (KeyboardInterrupt, callback failure) that would orphan `tool_use` blocks falls through to stub `tool_result`s, preserving the API's block-pairing invariant.
- Context pruning: old tool results are replaced with a placeholder when input tokens exceed `PRUNE_CONTEXT_THRESHOLD` of the model's max context. Logic lives in `context.py`.

## Conventions

- **Python 3.12.10**, pyenv virtualenv `python-agent`
- **Tool names:** `verb_noun` snake_case, regex-checked at `Agent.__init__`
- **`ToolName` StrEnum:** single source of truth for tool names
- **No code in `__init__.py`** — re-exports only
- **Shared tool utilities** live in `tools/_helpers.py`: `truncate()`, `clamp_timeout()`, `validate_file_path()`, `is_binary()`. Use them instead of reimplementing.

## Built-in Tools

- **File I/O:** `read_file`, `write_file`, `edit_file` (supports `replace_all` for bulk replacements)
- **Search:** `glob_files`, `grep_files`
- **Execution:** `run_bash`, `run_python`
- **Web:** `fetch_url` (HTTP GET → markdown/text/HTML via `httpx` + `markdownify`)
- **Agent control:** `make_plan`, `send_response` (intercepted)

## Adding a Tool

1. Create `python_agent/tools/your_tool.py` with a class satisfying the `Tool` Protocol.
2. Add `YOUR_TOOL = "your_tool"` to `ToolName` in `tools/base.py`.
3. Register in `tools/registry.py`.
4. Add a summarizer entry in `tool_summaries.py` `SUMMARIZERS` (required — without it, the fallback truncates params to 100 chars).
5. Optionally update WORKFLOW rule 2 in `prompts.py` if the tool replaces a `run_bash` pattern.

## Refactoring

When assessing refactoring opportunities, scan for: DRY violations, SRP violations, dead code, inconsistent patterns, functions exceeding ~40 lines, and violations of any Good Coding Principle. Only refactor what genuinely reduces duplication or fixes a real problem — never for style preferences or premature optimization.

## Pre-commit policy (MANDATORY)

Before every commit, in order:

1. **Re-run the Self-Review Checklist** on every modified file — correct, simple, elegant, well-written, maintainable, no dead code, principle-compliant.
2. **Scan for refactoring opportunities** using the list in the Refactoring section. Apply any that genuinely improve the touched code; defer the rest.
3. **Write tests** for every new behavior introduced by the commit (and for any untested behavior uncovered while working). Tests live in `tests/` mirroring the package layout.
4. **Run the full test suite** — all tests must pass before committing. Fix failures; never commit with broken tests.

No commit is exempt, regardless of size.

## Provider & Run

`API_PROVIDER` in `.env`: `anthropic` (default, needs `ANTHROPIC_API_KEY`) or `bedrock` (AWS creds + ARN inference-profile model IDs).

```bash
python -m python_agent
```

Launches the Textual TUI. Requires a real TTY (fails on piped stdin).
