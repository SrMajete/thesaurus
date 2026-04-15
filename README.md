# Python Agent

A streaming, tool-calling AI agent built on the Anthropic Claude API. Learning lab for agentic AI patterns — minimal core, easy to extend, three terminal UIs.

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
python -m python_agent                     # rich UI (default)
CLI_STYLE=classic python -m python_agent   # hand-rolled ANSI UI
CLI_STYLE=tui     python -m python_agent   # full-screen Textual TUI
```

## What it does

- Streams model output live (thinking, tool calls, response) with character-level rendering.
- Runs tools in parallel where safe, prompts for permission on anything state-changing.
- Shows a two-field plan (thinking + roadmap) that the model evolves every turn.
- TUI surfaces per-turn and cumulative token usage with a cache-hit breakdown.
- Supports direct Anthropic API and AWS Bedrock.

Built-in tools: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`, `run_bash`, `run_python`, plus the intercepted `make_plan` / `send_response` that drive the agent loop.

## Repo layout

```
python_agent/
├── cli.py           classic ANSI UI
├── cli_rich.py      Rich-library UI (default)
├── cli_tui.py       Textual full-screen TUI
├── client.py        AsyncAnthropic | AsyncAnthropicBedrock factory
├── tool_summaries.py  per-tool param previews shared by all three UIs
├── agent.py         conversation state + callback contract
├── processor.py     agentic loop (tool dispatch, streaming)
├── api_client.py    streaming Claude API call + jiter partial-JSON
├── prompts.py       system prompt + code-quality spec
├── config.py        pydantic-settings loader
└── tools/           one class per tool
```

## More

See [CLAUDE.md](CLAUDE.md) for the architecture, design decisions, conventions, and the mandatory self-review checklist.
