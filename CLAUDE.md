# Python Agent

A streaming AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns — modular, expandable, easy to add tools and techniques.

## Architecture

```
cli.py (I/O) → agent.py (state) → processor.py (logic) → api_client.py (API)
                                       ↕
                                  tools/*.py (execution)
```

- **Agent** owns state (messages, plan, tools, callbacks). One instance per session.
- **Processor** owns the loop logic. Pure — no I/O, no state. Reads/writes the Agent instance.
- **API client** handles streaming with jiter partial JSON parsing. Streams `make_plan.thinking`, `make_plan.roadmap`, and `send_response.response` live.
- **Tools** satisfy a `Tool` Protocol with class-level attributes. No inheritance. Adding a tool: one class, one `ToolName` enum entry, one registry line.
- **Callbacks** (`AgentCallbacks` dataclass) decouple the agent from terminal I/O.

## Quality Standards

All code must follow good coding and architecture principles: KISS, DRY, YAGNI, SRP, OCP, LSP, ISP, DIP, Separation of Concerns, Law of Demeter, Composition over Inheritance, Principle of Least Astonishment, Fail Fast, Boy Scout Rule, Encapsulation, and High Cohesion / Loose Coupling. See `prompts.py` CODE_QUALITY section for the full definitions. These apply to every change — code, finding, recommendation, or plan.

## Key Design Decisions

- `tool_choice: any` — every API response must use a tool. No freeform text.
- `make_plan` + `send_response` are intercepted tools — their `execute()` is never called. The processor handles them specially (persist plan, stream response).
- `agent.plan` is injected into the system prompt every turn via `_current_plan_section()`. The model sees its own thinking + roadmap and evolves it.
- Two-phase tool dispatch: `_collect_decisions` (batch permissions upfront) → `_execute_decisions` (parallel or serial execution).
- `needs_permission` × `is_parallelizable` flags produce four dispatch quadrants.
- `reason` field auto-injected as first property of every tool schema so the model writes intent before payload, enabling streaming tool headers.
- Prompt caching: static system sections in one cached block, dynamic plan uncached, tool definitions cached via last-tool `cache_control`, last user message gets ephemeral breakpoint.

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

## Running

```bash
cd python-agent
python -m python_agent
```

## Adding a Tool

1. Create `python_agent/tools/your_tool.py` with a class satisfying the `Tool` Protocol
2. Add `YOUR_TOOL = "your_tool"` to `ToolName` in `tools/base.py`
3. Register in `tools/registry.py`
4. Add a summarizer in `cli.py` `_SUMMARIZERS` dict
5. Update WORKFLOW rule 2 in `prompts.py` if the tool replaces a `run_bash` pattern

## Reference Code

The original TypeScript Claude Code CLI lives at `../original_code/`. Use it to check how the original implements a feature before building the Python equivalent — but write fresh idiomatic Python, don't port the TypeScript.

## Known Issues

- No context management — long sessions will hit the context window limit
- KeyboardInterrupt during tool execution leaves orphaned `tool_use` blocks (corrupts conversation state)
- Terminal output can interleave during parallel auto-approved tools (cosmetic, documented in processor.py)
- `_fit` in cli.py measures `len(text)` not terminal columns (emoji/CJK off by one)
