# OpenAI-Compatible Provider — Implementation Plan

Add support for any OpenAI-compatible API (ollama, vLLM, llama.cpp, LM Studio,
OpenAI, Groq, Together, Fireworks, DeepSeek, etc.) as a new `LLMClient` adapter.

The OpenAI chat completions protocol is the de facto universal standard — one
adapter covers every local and remote engine that speaks it.

---

## Files to change

### 1. `adapters/config.py` — extend Settings

```python
api_provider: Literal["anthropic", "bedrock", "openai"] = "anthropic"

# OpenAI-compatible provider (used when api_provider == "openai")
openai_base_url: str = "http://localhost:11434/v1"  # ollama default
openai_api_key: str = "ollama"  # most local backends accept anything
```

Update `_validate_provider_settings`: skip the `ANTHROPIC_API_KEY` requirement
when `api_provider == "openai"`.

### 2. `adapters/client_factory.py` — add factory branch

In `make_llm_client`:

```python
elif settings.api_provider == "openai":
    from .openai_llm import OpenAILLMClient
    return OpenAILLMClient(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.model,
    )
```

In `fetch_max_context_tokens`: return `_DEFAULT_MAX_CONTEXT_TOKENS` for the
`openai` path (same as bedrock — no introspection endpoint available).

### 3. `adapters/openai_llm.py` — new file (~150 lines)

The only real work. A new `OpenAILLMClient` class implementing the `LLMClient`
protocol. Wraps `openai.AsyncOpenAI` the same way `AnthropicLLMClient` wraps
`anthropic.AsyncAnthropic`.

### 4. `requirements.txt` — one new dependency

```
openai>=1.0.0
```

---

## Adapter responsibilities

The adapter is a pure format translator at the boundary:

```
Anthropic world (internal)  ←→  OpenAI world (wire)
```

### a) Tool schema translation (once, at `__init__`)

Anthropic format:
```json
{"name": "...", "description": "...", "input_schema": {...}}
```

OpenAI format:
```json
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```

A one-liner dict comprehension.

### b) Message history translation (per call)

| Anthropic (internal)                     | OpenAI (wire)                                      |
|------------------------------------------|----------------------------------------------------|
| `system` blocks list                     | `{"role": "system", "content": "..."}` at index 0  |
| `{"role": "user", "content": [...]}`     | `{"role": "user", "content": "..."}`               |
| `{"type": "text", "text": "..."}`        | `{"role": "assistant", "content": "..."}`           |
| `{"type": "tool_use", id, name, input}`  | `{"role": "assistant", "tool_calls": [...]}`        |
| `{"type": "tool_result", ...}`           | `{"role": "tool", "tool_call_id": "...", ...}`      |

Key detail: Anthropic assistant messages can have multiple content blocks
(text + tool_use mixed). These must be split or merged into the OpenAI
shape where `tool_calls` is a top-level array on the assistant message and
text lives in `content`.

### c) Request translation (per call)

| Anthropic                             | OpenAI                       |
|---------------------------------------|------------------------------|
| `tool_choice: {"type": "any"}`        | `tool_choice: "required"`    |
| `max_tokens: 16384`                   | `max_tokens: 16384`          |
| `system` as separate parameter        | `system` as first message    |

### d) Response translation (per call)

Accumulate streaming chunks from the OpenAI SDK, then build:

- `AgentResponse.content` — list of `{"type": "text", ...}` and
  `{"type": "tool_use", ...}` dicts in the Anthropic shape
- `AgentResponse.tool_calls` — list of `ToolCall(id, name, input)` objects
- `AgentResponse.stop_reason` — map OpenAI's `finish_reason` to Anthropic's
  `stop_reason` values (`"stop"` → `"end_turn"`, `"tool_calls"` → `"tool_use"`)
- Token counts from `usage` in the final chunk; cache fields set to `0`
- Tool use IDs generated via `uuid4()` (Anthropic generates these server-side)

---

## Streaming strategy

### Phase 1: batch mode (start here)

Accumulate the full response, then fire all callbacks at once:

```python
# After stream completes:
for tool_call in parsed_tool_calls:
    on_tool_start(tool_call.id, tool_call.name, tool_call.input)
on_text(text_content)
return AgentResponse(...)
```

The TUI works — responses appear as a batch instead of character-by-character.
The pre-stream "thinking…" spinner bridges the wait.

### Phase 2: live streaming (later)

Port the jiter-based partial JSON parsing from `anthropic_llm.py` to work with
OpenAI's streaming format (`chunk.choices[0].delta.tool_calls[N].function.arguments`).
Same concept, different event shape. This makes `make_plan` thinking and
`send_response` text stream live to the TUI.

---

## Token counting

| Field                          | Source                                         |
|--------------------------------|------------------------------------------------|
| `input_tokens`                 | `usage.prompt_tokens` from final chunk         |
| `output_tokens`                | `usage.completion_tokens` from final chunk     |
| `cache_creation_input_tokens`  | `0` (no prompt caching on local models)        |
| `cache_read_input_tokens`      | `0`                                            |

Most backends (ollama, vLLM) include `usage` in streaming responses when
`stream_options={"include_usage": True}` is set. If a backend omits it,
fall back to `0` — the TUI counter shows zeros and context pruning uses
the default threshold, which is safe.

---

## Configuration examples

### Ollama (local)

```env
API_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
MODEL=qwen2.5-coder:32b
```

### vLLM (local)

```env
API_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=token-abc123
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

### OpenAI (remote)

```env
API_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
MODEL=gpt-4o
```

### Groq (remote)

```env
API_PROVIDER=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
MODEL=llama-3.3-70b-versatile
```

---

## Caveats and known limitations

### Model quality

The agent loop is demanding: every turn must produce a valid tool call
(`tool_choice: required`), the system prompt mandates `make_plan` before
any action, and tool inputs must be well-formed JSON matching specific schemas.

Recommended models for reliable tool use:
- **Qwen 2.5 Coder 32B** — best open coding model with function calling
- **Llama 3.3 70B** — strong general-purpose with tool support
- Anything smaller will likely produce malformed tool calls or infinite loops

### Context window

Most local models cap at 32K–128K tokens vs Claude's 200K. The existing
pruning logic (`core/context.py`) handles this, but `prune_context_threshold`
may need lowering (e.g. `0.6` instead of `0.8`) to trigger earlier.

### Retry on malformed output

Unlike Claude, local models occasionally produce invalid JSON in tool call
arguments. The adapter should catch `json.JSONDecodeError` on tool argument
parsing and retry once (or return an error `AgentResponse` that the processor
can handle gracefully). The Anthropic adapter never needs this.

### The `make_plan` discipline

Smaller models may not reliably follow the "always call `make_plan` first"
instruction. Future work could make plan enforcement optional via a config
flag, relaxing the system prompt for providers where models struggle with it.

---

## What stays untouched

- `core/agent.py` — protocol-typed, accepts any `LLMClient`
- `core/processor.py` — works against `AgentResponse`/`ToolCall`, no SDK knowledge
- `core/ports.py` — the Protocol definition; the new adapter satisfies it
- `tools/*` — completely decoupled from inference
- `adapters/tui.py` — works against callbacks, doesn't know about the LLM
- `adapters/anthropic_llm.py` — unchanged, still the default

The hexagonal architecture confines this change to one new file and two
small edits to existing files.
