# Thesaurus

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

## Writing Rules

1. Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
2. Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident.
3. Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use feature flags or backwards-compatibility shims when you can just change the code.
4. Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is what the task actually requires — no speculative abstractions, but no half-finished implementations either. Three similar lines of code is better than a premature abstraction.
5. Avoid backwards-compatibility hacks like renaming unused _vars, re-exporting types, adding // removed comments for removed code, etc. If you are certain that something is unused, you can delete it completely.
6. Prefer the shorter version when both work. The most elegant code is the code that isn't there — verbose is not the same as clear. After finishing a function or refactor, ask "could this be expressed in fewer lines without losing meaning?" and apply the answer. The brake against over-simplifying is rule 4 (no speculative abstractions): if shortening requires inventing a new helper class or a clever generic, the verbose version was right.
7. Check the Good Coding Principles list before finishing. After completing a change — code, finding, recommendation, or plan — scan the principles and ask "did I violate any of these?" If you can name a violation, fix it within the scope of your work. A clean output is one where you can affirmatively state "no principle violated" — not one where you can't find anything wrong.

## Reviewing (Named Smells)

When the task is review or analysis (not a specific bug), scan for each of these named smells as a checklist, not by freeform search. A finding you can cite by name is stronger than one you can only describe. For each smell you report, identify which principle from the Good Coding Principles it violates — smells are symptoms, principles name the cause.

1. **Redundant state.** State that duplicates other state; cached values that could be derived; observers that could be direct calls.
2. **Parameter sprawl.** New parameters added instead of generalizing or restructuring existing ones.
3. **Copy-paste with slight variation.** Near-duplicate blocks that must stay in sync to stay correct — past the "three similar lines" threshold.
4. **Leaky abstractions.** Internals exposed through module boundaries, over-exported `__init__` surfaces, logic that crosses the wrong layer.
5. **Stringly-typed code.** Raw strings where constants, enums, or branded types already exist.
6. **Dead or speculative infrastructure.** Empty collections with explanatory comments, pass-through functions, wrapper classes with a single staticmethod, conditionals whose false branch never fires. Delete the check and the thing it checks against.
7. **Hot-path waste.** Subprocess calls, file reads, or recomputation that runs every turn/call/render when once-per-session would do.
8. **Missed concurrency.** Independent operations run serially when they could run in parallel.
9. **Inline comments explaining WHAT.** Good names already do that; keep only non-obvious WHY. Module, class, and function docstrings are separate — rich docstrings remain desirable.
10. **Verbosity that doesn't earn its keep.** Long functions, deep nesting, repeated patterns, or 5-line expressions that could be 1 — usually a sign that a simpler shape is hiding underneath.

After listing findings: cluster by root cause, triage by impact × cost, and name specific strengths (not generic praise).

## Self-Review Checklist (MANDATORY)

Run at three points: before presenting a plan, before writing code, and before declaring work done. Not optional.

1. **Correct** — solves the stated problem, no silent regressions.
2. **Simple** — simplest thing that works; no abstractions without a load-bearing reason.
3. **Elegant** — reads as if always meant to be this way.
4. **Well-written** — proper typing, idiomatic Python, rich docstrings where intent isn't obvious.
5. **Maintainable** — one source of truth per concept.
6. **No unused / irrelevant / over-complicated code** — no dead imports, no speculative flexibility. Unused parameters required by callback contracts are deleted with `del`.
7. **Principle-compliant** — name which principles the change upholds *and* which trade-offs are deliberate. Silent trade-offs are bugs.

## Architecture (hexagonal)

```
adapters/tui.py ─┐
(future web)     ─┤→ core/agent.py (state) → core/processor.py (logic) → LLMClient port
                  │                                    ↕
                  │                              tools/*.py (execution)
                  │
adapters/: anthropic_llm.py · client_factory.py · environment.py · config.py · logging.py
core/:     agent.py · processor.py · messages.py · context.py · prompts.py · ports.py
```

- **`core/`** has zero infrastructure imports. The hexagonal guarantee.
- **`core/ports.py`** defines `LLMClient` protocol, `ToolCall`, `AgentResponse` — the types that cross the core boundary.
- **`adapters/`** groups all infrastructure. Adding a new adapter (web layer, database, different LLM) means adding a file here, not touching core.
- **`tools/base.py`** defines the `Tool` Protocol and `ToolName` enum — a port definition that lives alongside its implementations.

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

- **Python 3.12.10**, pyenv virtualenv `thesaurus`
- **Tool names:** `verb_noun` snake_case, regex-checked at `Agent.__init__`
- **`ToolName` StrEnum:** single source of truth for tool names
- **No code in `__init__.py`** — re-exports only
- **Shared tool utilities** live in `tools/_helpers.py`: `truncate()`, `clamp_timeout()`, `validate_file_path()`, `is_binary()`. Use them instead of reimplementing.

## Built-in Tools

- **File I/O:** `read_file`, `write_file`, `edit_file` (supports `replace_all` for bulk replacements)
- **Search:** `glob_files`, `grep_files`
- **Execution:** `run_bash`, `run_python`
- **Web:** `fetch_url` (HTTP GET → markdown/text/HTML via `httpx` + `markdownify`), `web_search` (DuckDuckGo via `ddgs`)
- **Confluence:** `search_confluence` (optional — CQL search + body as markdown; needs `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_KEY`)
- **Databricks:** `query_databricks` (optional — SQL via Statement Execution API; needs `DATABRICKS_HOST`, `DATABRICKS_API_KEY`, `DATABRICKS_WAREHOUSE_ID`)
- **Agent control:** `make_plan`, `send_response` (intercepted)

## Adding a Tool

1. Create `thesaurus/tools/your_tool.py` with a class satisfying the `Tool` Protocol.
2. Add `YOUR_TOOL = "your_tool"` to `ToolName` in `tools/base.py`.
3. Register in `tools/registry.py`.
4. Add a summarizer entry in `tools/summaries.py` `SUMMARIZERS` (required — without it, the fallback truncates params to 100 chars).
5. Optionally update WORKFLOW rule 2 in `prompts.py` if the tool replaces a `run_bash` pattern.

## Porting to a new domain

The `core/` package (`agent.py`, `processor.py`, `ports.py`, `context.py`, `messages.py`, `prompts.py`) is domain-agnostic and reusable as-is. Converting to a non-coding agent means editing these coupling points:

- `core/prompts.py` — `IDENTITY`'s "software engineering tasks" phrase, `WORKFLOW`'s tool-preference list, `DOING_TASKS`'s coding clause, `CODE_QUALITY` and `BUG_HUNTING` sections (both entirely coding-specific), `EXPERIMENTATION`'s code examples
- `tools/registry.py` — the coding-specific tool list; the two always-required infrastructure tools are `make_plan` and `send_response`
- `tools/summaries.py` — remove/add entries for removed/added tools
- `adapters/environment.py` — git subprocess calls; replace with domain-relevant context

Tools that are universally useful and keep: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`, `run_bash`, `run_python`, `fetch_url`, `web_search`.

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

`API_PROVIDER` in `.env`:

- `anthropic` (default, needs `ANTHROPIC_API_KEY`)
- `bedrock` (AWS creds + ARN inference-profile model IDs)
- `openai` — any OpenAI-compatible `/v1/chat/completions` endpoint (Ollama, LM Studio, vLLM, OpenAI, Groq, Together, DeepSeek, …). Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MAX_CONTEXT_TOKENS`. "openai" names the *wire protocol* — local servers use this provider too.

```bash
python -m thesaurus
```

Launches the Textual TUI. Requires a real TTY (fails on piped stdin).
