# Thesaurus

A streaming, tool-calling AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns — minimal core, easy to extend, Textual-based TUI.

## Quick start

```bash
# 1. Python 3.12.10 + pyenv virtualenv
pyenv virtualenv 3.12.10 thesaurus
pyenv activate thesaurus
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY (or switch to AWS Bedrock)

# 3. Run
python -m thesaurus
```

## What it does

- Streams model output live (thinking, tool calls, response) with character-level rendering.
- Runs tools in parallel where safe, prompts for permission inline on anything state-changing.
- Shows a two-field plan (thinking + roadmap) that the model evolves every turn.
- Surfaces per-turn and cumulative token usage with a cache-hit breakdown.
- Queues user prompts typed during a running turn — executed in submission order.
- Supports direct Anthropic API and AWS Bedrock.

Built-in tools: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`, `run_bash`, `run_python`, `fetch_url`, `web_search`, `search_confluence` (optional, needs config), plus the intercepted `make_plan` / `send_response` that drive the agent loop.

## Repo layout (hexagonal)

```
thesaurus/
├── core/                  the hexagon — zero infrastructure imports
│   ├── agent.py           conversation state + AgentCallbacks port
│   ├── processor.py       agentic loop (calls LLMClient port)
│   ├── ports.py           LLMClient protocol, ToolCall, AgentResponse
│   ├── prompts.py         system prompt assembly (pure strings)
│   ├── messages.py        message factories
│   └── context.py         token pruning
├── adapters/              infrastructure implementations
│   ├── anthropic_llm.py   AnthropicLLMClient (LLMClient adapter)
│   ├── client_factory.py  make_llm_client from Settings
│   ├── environment.py     OS/git env info (injected into Agent)
│   ├── config.py          pydantic-settings loader
│   ├── logging.py         logging setup
│   └── tui.py             Textual full-screen TUI
└── tools/                 one class per tool (Tool Protocol)
    └── summaries.py       per-tool display metadata
```

## More

See [CLAUDE.md](CLAUDE.md) for the architecture, design decisions, conventions, and the mandatory self-review checklist.
