# Python Agent

A streaming, tool-calling AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns — minimal core, easy to extend, Textual-based TUI.

## Quick start

```bash
# 1. Python 3.12.10 + pyenv virtualenv
pyenv virtualenv 3.12.10 python-agent
pyenv activate python-agent
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY (or switch to AWS Bedrock)

# 3. Run
python -m python_agent
```

## What it does

- Streams model output live (thinking, tool calls, response) with character-level rendering.
- Runs tools in parallel where safe, prompts for permission inline on anything state-changing.
- Shows a two-field plan (thinking + roadmap) that the model evolves every turn.
- Surfaces per-turn and cumulative token usage with a cache-hit breakdown.
- Queues user prompts typed during a running turn — executed in submission order.
- Supports direct Anthropic API and AWS Bedrock.

Built-in tools: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`, `run_bash`, `run_python`, `fetch_url`, `web_search`, `search_confluence` (optional, needs config), plus the intercepted `make_plan` / `send_response` that drive the agent loop.

## Repo layout

```
python_agent/
├── cli_tui.py       Textual full-screen TUI (the only frontend)
├── client.py        AsyncAnthropic | AsyncAnthropicBedrock factory
├── tool_summaries.py  per-tool param previews
├── agent.py         conversation state + callback contract
├── processor.py     agentic loop (tool dispatch, streaming)
├── api_client.py    streaming Claude API call + jiter partial-JSON
├── prompts.py       system prompt + code-quality spec
├── config.py        pydantic-settings loader
└── tools/           one class per tool
```

## More

See [CLAUDE.md](CLAUDE.md) for the architecture, design decisions, conventions, and the mandatory self-review checklist.
