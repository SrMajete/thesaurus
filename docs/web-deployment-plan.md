# Deployment roadmap — multi-user web service for thousands of concurrent customers

## Context

The `python_agent/` core loop is already concurrency-safe and streaming-capable. This plan covers everything needed to wrap it in a production web service: FastAPI + Gunicorn + Postgres + Docker + session checkpointing. Scoped for **thousands of concurrent customers**.

**Locked decisions (from prior discussion):**

- **Stack**: FastAPI, Uvicorn workers under Gunicorn, Postgres, Docker multi-stage
- **Streaming**: SSE + REST (no WebSocket — tools are safe, no permission prompts)
- **Auth**: AWS IAM for Bedrock (server-side), JWT for end-user identity
- **Agent framework**: our custom loop — **no LangGraph**
- **Storage**: Postgres JSONB with normalized `sessions` + `messages` tables (append-only), **no custom binary format**
- **Checkpoints**: two writes per turn — on user message receive, on `send_response`
- **Tool scope**: customer-facing profile is a restricted safe set (RAG, domain lookups, troubleshoot, fetch_url, web_search). `run_bash`/`run_python`/`write_file`/`edit_file` removed from web profile.

**Architecture at a glance:**

```
client (browser / app)
    │  HTTPS / SSE
    ▼
load balancer
    │
    ▼
Gunicorn (N workers) — Uvicorn worker class, own DB pool per worker
    │
    ├─► in-memory session cache (per worker, TTL eviction)
    ├─► Postgres — sessions + messages (JSONB, append-only)
    ├─► Bedrock — Claude via AWS IAM
    └─► S3 — archive + externalized large tool outputs (later)
```

---

## Phase 1 — Foundation (~1 week solo)

**Goal**: end-to-end system in Docker Compose. One user, one session, streaming replies, state persists across restarts.

### Step 1.1 — Add `on_checkpoint` callback to the agent

**Files**: `python_agent/agent.py`, `python_agent/processor.py`, `tests/test_agent.py`, `tests/test_processor.py`

- Add `on_checkpoint: Callable[[], Awaitable[None]]` to `AgentCallbacks`
- In `processor.run_loop`, call `await agent.callbacks.on_checkpoint()` at two points:
  1. Immediately after the user message is appended (before the first `stream_response`)
  2. In the `send_response` exit path, right before `return`
- TUI provides a no-op implementation: `async def _noop(): pass`
- Update existing tests to pass the no-op callback

**Exit criterion**: existing 281 tests pass; new unit test verifies the callback fires twice per turn.

### Step 1.2 — Create Postgres schema

**Files**: `_ci/migrations/001_initial.sql`

```sql
CREATE TABLE sessions (
    id                    UUID PRIMARY KEY,
    user_id               UUID NOT NULL,
    profile               TEXT NOT NULL DEFAULT 'customer_service',
    title                 TEXT,
    plan                  TEXT,
    total_input_tokens    BIGINT NOT NULL DEFAULT 0,
    total_output_tokens   BIGINT NOT NULL DEFAULT 0,
    last_context_tokens   INTEGER NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at           TIMESTAMPTZ
);
CREATE INDEX idx_sessions_user ON sessions (user_id, last_activity_at DESC)
    WHERE archived_at IS NULL;

CREATE TABLE messages (
    id            UUID PRIMARY KEY,
    session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       JSONB NOT NULL,
    tokens        INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_messages_session_seq ON messages (session_id, seq);
```

**Exit criterion**: schema applies cleanly to a fresh Postgres instance.

### Step 1.3 — Minimal profile seam for tools

**Files**: `python_agent/tools/registry.py`

- `get_default_tools(profile: str = "coding") -> list[Tool]`
- `profile="coding"` (default) returns today's full list — no existing callers break
- `profile="customer_service"` returns safe subset: `MakePlanTool`, `SendResponseTool`, `FetchUrlTool`, `WebSearchTool` (domain-specific tools stubbed for now; implement in Phase 4)

**Exit criterion**: existing TUI still works; new profile returns the expected subset.

### Step 1.4 — Create the web package skeleton

**Files**: `python_agent_web/__init__.py`, `python_agent_web/app.py`, `python_agent_web/db.py`

- `app.py`: FastAPI app with lifespan hook (opens psycopg pool on startup, closes on shutdown)
- `db.py`: `AsyncConnectionPool` wrapper, `execute`, `fetch_one`, `fetch_all` helpers using parameterized queries
- Pool sized per-worker: `min=2`, `max=10`, configurable via env

**Exit criterion**: `uvicorn python_agent_web.app:app` starts and connects to Postgres.

### Step 1.5 — Session store and rehydration

**Files**: `python_agent_web/session.py`, `python_agent_web/callbacks.py`

- `SessionStore.create(user_id, profile) -> session_id` — INSERTs into `sessions`, returns UUID
- `SessionStore.get(session_id) -> Agent` — looks up in-memory cache; if miss, reads `sessions` + `messages` from DB and builds a fresh `Agent` with messages populated
- `SessionStore.evict(session_id)` — TTL-based eviction (10 min idle), agent released for GC
- `WebSessionCallbacks` implements `AgentCallbacks`:
  - `on_text`, `on_thinking`, `on_tool_start`, `on_tool_result` → push onto a per-session asyncio.Queue consumed by the SSE endpoint
  - `ask_permission` → returns `True` immediately (auto-approve, tools are safe)
  - `on_checkpoint` → writes new messages to DB (append-only INSERT) + updates `sessions` row with token counters and plan

**Exit criterion**: unit tests verify rehydration produces an `Agent` with the same message state as before restart.

### Step 1.6 — REST + SSE endpoints

**Files**: `python_agent_web/routes/chat.py`, `python_agent_web/routes/sessions.py`, `python_agent_web/routes/health.py`, `python_agent_web/schemas.py`

- `POST /chat/stream` → input `{session_id, text}`, returns `StreamingResponse` with `text/event-stream`. Consumer reads from the session's queue, emits SSE events. Events: `text`, `thinking`, `tool_start`, `tool_result`, `end`, `error`.
- `POST /chat/message` → same input, waits for end-of-turn, returns full response as JSON. For non-streaming clients and testing.
- `POST /sessions` → creates session, returns `{session_id}` (Phase 1 takes `user_id` in body; Phase 2 derives from JWT).
- `GET /chat/{session_id}/messages` → returns conversation history from DB.
- `DELETE /chat/{session_id}` → soft-delete (`archived_at = NOW()`).
- `GET /health` → DB ping, returns 200 if OK.

**Exit criterion**: curl can create a session, stream a turn, fetch history, delete.

### Step 1.7 — Dockerfile (multi-stage)

**Files**: `Dockerfile`, `.dockerignore`

- **Builder stage**: `python:3.12-slim` + Rust toolchain + build deps. Install requirements (compiles `primp` wheel).
- **Runtime stage**: `python:3.12-slim`, no Rust. Copy installed packages from builder. App code. Non-root user. Entrypoint `/run.sh` which runs Gunicorn.
- `.dockerignore` excludes tests, docs, local virtualenv.

**Exit criterion**: `docker build -t agent-web .` produces an image; image boots.

### Step 1.8 — docker-compose for local dev

**Files**: `docker-compose.yml`, `_ci/run.sh`

- Services: `web` (our image), `postgres:15`
- Volume for Postgres data persistence between restarts
- `web` depends on `postgres` (healthcheck)
- Migration runs on container startup via `run.sh`: `psql -f _ci/migrations/001_initial.sql` then Gunicorn

**Exit criterion**: `docker compose up` brings the stack up; curl works against `localhost:8000`.

### Step 1.9 — Tests for the web layer

**Files**: `tests/web/test_sessions.py`, `tests/web/test_chat_endpoints.py`, `tests/web/test_rehydration.py`, `tests/web/conftest.py`

- `conftest.py`: pytest fixture for a test Postgres (testcontainers-python), fresh DB per test
- Unit tests: SSE event serialization, session rehydration, DB write failure handling
- Integration test: start web app in-process, exercise the full flow (create session → send message → verify DB state → restart → resume)
- Coverage target: ≥99% on `python_agent_web/` (same bar as core agent)

**Exit criterion**: all tests green, coverage target met.

### Phase 1 exit criteria

- `docker compose up`, curl creates a session, streams a response, kills the web container, restarts, resumes the conversation intact
- Existing 281 core-agent tests + new web-layer tests all pass
- Can be demoed end-to-end

---

## Phase 2 — Production hardening (~2 weeks solo)

**Goal**: safe to point real users at. Auth, error handling, observability, graceful operations, production Gunicorn config.

### Step 2.1 — JWT authentication middleware

**Files**: `python_agent_web/auth.py`, `python_agent_web/app.py`

- FastAPI dependency: extracts bearer token, validates against issuer's public key (JWKS), returns `user_id`
- Middleware applied to all `/chat/*` and `/sessions/*` routes
- `SessionStore.get` checks `sessions.user_id == request.user_id`; 403 if mismatch
- `Settings` adds `jwt_issuer`, `jwt_audience`, `jwt_jwks_url`

**Exit criterion**: unauthenticated requests get 401; cross-user access gets 403; valid JWT works.

**Blocking question for this step**: which auth service issues the JWT? If none exists, we need to either build one (out of scope) or use a signed-cookie shim with a static server key for pre-production.

### Step 2.2 — Bedrock retry/fallback on transient errors

**Files**: `python_agent/api_client.py` (minimal change) or a wrapper in the web layer

- On 429 (throttling): retry up to 3 times with exponential backoff (1s, 2s, 4s)
- On 500/503: retry once after 1s
- On retry exhausted: raise a typed exception the web layer catches → emits `event: error` on SSE, leaves session intact
- Add structured log entries for each retry attempt

**Exit criterion**: simulated 429 gets retried; simulated repeated 500 fails cleanly without crashing the worker.

### Step 2.3 — Per-user rate limiting

**Files**: `python_agent_web/rate_limit.py`, `_ci/migrations/002_rate_limits.sql`

- New table `rate_limits(user_id, window_start, request_count, token_count)`
- Dependency `check_rate_limit(user_id)` runs before each `/chat/*` request
- Limits configurable via env: requests/min, tokens/day
- On limit exceeded: 429 with `Retry-After` header and error JSON
- Postgres-backed for simplicity; if contention appears in load tests, migrate to Redis in Phase 3

**Exit criterion**: test hits the per-minute limit → gets 429 → waits → succeeds.

### Step 2.4 — Structured logging

**Files**: `python_agent/logging_config.py` (updates), `python_agent_web/app.py`

- All logs JSON-formatted via `python-json-logger` or equivalent
- Correlation IDs: `request_id` (generated per request), `session_id`, `user_id` propagated via `contextvars`
- No `print`, no plain-format log lines in any new code
- Log levels per logger name configurable via env

**Exit criterion**: log lines parse cleanly with `jq`; each log line carries the correlation IDs for the active request.

### Step 2.5 — OpenTelemetry traces and metrics

**Files**: `python_agent_web/observability.py`, `python_agent_web/app.py`

- OTel SDK initialization in lifespan hook
- Auto-instrument FastAPI (via `opentelemetry-instrumentation-fastapi`) and `psycopg`
- Manual spans around: agent turn, each tool call, Bedrock API call
- Metrics: active_sessions gauge, turns_per_second counter, turn_latency_ms histogram, bedrock_input_tokens counter, bedrock_output_tokens counter
- Exporter configurable via env (start with stdout exporter; Phase 3 wires to your org's backend)

**Exit criterion**: a full turn produces a trace with nested spans; metrics scrape endpoint (or OTel collector output) shows non-zero counters.

### Step 2.6 — Graceful shutdown

**Files**: `python_agent_web/app.py`, Gunicorn config

- SIGTERM handler stops accepting new connections
- In-flight SSE streams allowed to finish (with a 30s hard cap)
- DB pool closed after last stream completes
- Gunicorn `graceful_timeout = 60`, `timeout = 180`

**Exit criterion**: `docker kill -s SIGTERM` during an in-flight stream → stream completes → container exits cleanly within 60s.

### Step 2.7 — Production Gunicorn config

**Files**: `_ci/gunicorn.conf.py`, `Dockerfile` (entrypoint)

- Workers = `int(os.getenv("WEB_WORKERS", (2 * os.cpu_count()) + 1))`
- Worker class: `uvicorn.workers.UvicornWorker`
- `preload_app = False` — each worker opens its own DB pool (avoids fork + pool issues)
- `max_requests = 1000`, `max_requests_jitter = 50` — recycles workers to combat memory growth
- `worker_tmp_dir = "/dev/shm"` — speeds up worker process bookkeeping
- Log to stdout; container runtime (ECS/K8s) collects

**Exit criterion**: production config boots, handles load, workers recycle without dropping in-flight requests.

### Step 2.8 — CI pipeline

**Files**: `.gitlab-ci.yml` or `.github/workflows/ci.yml`

- Stages: lint (`ruff`), typecheck (`mypy`), test (`pytest --cov`), build Docker image, push to registry
- Coverage ≥99% gating
- Docker image tagged with commit SHA and branch name

**Exit criterion**: every commit triggers CI; failing tests or coverage drop block merge.

### Phase 2 exit criteria

- Can deploy to a staging environment behind a real load balancer
- JWT auth enforced; rate limits enforced; Bedrock outages degrade gracefully
- Traces and metrics flowing; structured logs parseable
- Graceful shutdown verified under load

---

## Phase 3 — Scale and operations (~2 weeks solo)

**Goal**: thousands of concurrent sessions. Retention. Cost controls. Multi-worker correctness validated.

### Step 3.1 — Alembic migrations

**Files**: `alembic/`, `alembic.ini`

- Initialize alembic
- Baseline migration matches the schema so far
- Migration-runner as a separate init container (K8s) or pre-deploy job — not part of web container startup
- Every schema change from here goes through alembic

**Exit criterion**: `alembic upgrade head` produces identical schema to the Phase 1 SQL file.

### Step 3.2 — Load test

**Files**: `_ci/load/load_test.py` (locust or k6)

- Scenario: 1000 concurrent SSE sessions, each sending one message every 30s
- Measure: p50/p95/p99 turn latency, DB connection pool saturation, Bedrock throttling rate, worker memory growth, worker CPU
- Run against staging with realistic Bedrock model
- Identify bottleneck (DB pool, Bedrock quota, CPU, etc.)

**Exit criterion**: documented baseline for 1000 concurrent sessions with target p99 latency. Bottleneck identified and tuning plan written.

### Step 3.3 — Address the bottleneck found in 3.2

Choose based on what the load test revealed:

- **If DB pool saturated**: increase pool size per worker, decrease worker count, or add pgBouncer in front of Postgres
- **If Bedrock throttling**: request Bedrock quota increase; implement queue backpressure
- **If memory growth**: reduce `max_requests_jitter`, investigate the leak; add memory metrics
- **If CPU-bound**: profile and optimize the hot path; scale horizontally

**Exit criterion**: rerun load test, meet target p99 latency.

### Step 3.4 — Retention and archival

**Files**: `python_agent_web/jobs/archive.py`, `_ci/cron/archive-cron.yaml` (K8s CronJob) or equivalent

- Daily job:
  1. `SELECT id FROM sessions WHERE last_activity_at < NOW() - INTERVAL '30 days' AND archived_at IS NULL`
  2. For each: fetch messages, write JSONL to `s3://bucket/archive/{user_id}/{session_id}.jsonl`
  3. `UPDATE sessions SET archived_at = NOW() WHERE id = ...`
- S3 lifecycle policy deletes archives after 1 year
- Configurable intervals: `ARCHIVE_AFTER_DAYS`, `DELETE_AFTER_DAYS`

**Exit criterion**: cron runs nightly in staging; verified active DB size bounded; archives readable from S3.

### Step 3.5 — Cost controls

**Files**: `python_agent_web/rate_limit.py` (extend), dashboard config

- Per-user monthly token budget; hard cap (429 when exceeded)
- Per-session max turns (`max_turns` from the existing agent config — surface as setting)
- Bedrock spend dashboard: aggregate tokens from the `sessions` row counters; render in your org's observability tool
- Alert: per-user daily spend > 10× median user → flag for review

**Exit criterion**: a user hitting the monthly cap gets a 429 with "quota exceeded" message; dashboard shows accurate per-hour / per-user spend.

### Step 3.6 — Audit logging

**Files**: `_ci/migrations/003_audit_log.sql`, `python_agent_web/audit.py`

- New table: `audit_log(id, user_id, session_id, action, metadata JSONB, created_at)`
- Events recorded: session created, session deleted, auth success, auth failure, rate limit hit, tool call (name + params sanitized)
- Retained 90 days minimum (configurable)

**Exit criterion**: auditing key events; queryable by user_id or session_id for support and compliance.

### Step 3.7 — Session cache strategy decision

**Files**: depends on outcome

By Phase 3 the in-memory per-worker cache has been measured in production.

- **If p99 is acceptable as-is**: keep it. Document the trade-off.
- **If worker-switch rehydration is noticeable**: configure the load balancer with sticky sessions on `session_id`.
- **If sticky sessions don't fit the deployment**: add Redis as a shared hot cache. Extra service, but correct at highest scale.

**Exit criterion**: decision recorded in `docs/` with evidence from load tests.

### Phase 3 exit criteria

- Load test of 1000 concurrent sessions passes target p99 latency
- Retention job runs nightly; active DB size bounded; archives verified
- Per-user quotas enforced; spend visible on dashboard
- Audit log captures the events required for compliance

---

## Phase 4 — Domain features (ongoing, product-driven)

**Goal**: the actual chatbot features on top of the platform.

Order-by-priority, loosely independent:

### Step 4.1 — RAG tool

- **Blocking decision**: which backend? pgvector (same DB, lower ops), Pinecone (managed, more $), OpenSearch (already in-house?), etc.
- Once chosen: ingestion pipeline (knowledge_chunks table or external index), embedding model (Titan via Bedrock is consistent with the stack), `rag_search(query, top_k)` tool, reindex triggers
- Implementation is ~1-2 weeks depending on pipeline complexity

### Step 4.2 — Domain-specific lookup tools

- Concrete list from product: `lookup_customer(id)`, `lookup_device(serial)`, etc.
- Each is a thin wrapper around existing REST/gRPC backends
- ~1 day per tool

### Step 4.3 — Troubleshooting flow

- Recommend starting tool-based: `start_troubleshooting(problem)` and `advance_troubleshooting(user_response)`, state lives in message history
- If analytics requirements surface later, upgrade to a structured `troubleshooting_sessions` table
- ~1 week for v1

### Step 4.4 — Feedback endpoint

- `POST /chat/{session_id}/feedback` with rating and comment
- New `feedback` table
- ~1 day

### Step 4.5 — Async session titles

- After first turn, background task calls a small Bedrock query ("title this conversation") and writes to `sessions.title`
- ~half a day

### Step 4.6 — Abuse detection

- Prompt-injection heuristics, per-user anomaly detection
- Non-trivial — scope separately

---

## Dependency graph between steps

```
1.1 on_checkpoint ────────┐
1.2 schema ───────────────┤
1.3 profile seam ─────────┤
                          ▼
                   1.4 web skeleton ──► 1.5 session store ──► 1.6 routes ──► 1.9 tests
                          │                                          │
                          └──► 1.7 Dockerfile ──► 1.8 compose ───────┘

Phase 2 depends on Phase 1 complete.
  2.1 JWT ─────────┐
  2.2 retries ─────┤
  2.3 rate limits ─┼──► 2.7 Gunicorn config ──► 2.8 CI
  2.4 logging ─────┤
  2.5 observability┤
  2.6 shutdown ────┘

Phase 3 depends on Phase 2 deployed somewhere:
  3.1 alembic ─► 3.2 load test ─► 3.3 fix bottleneck ─► 3.4 retention ─► 3.5 cost ─► 3.6 audit ─► 3.7 cache strategy
```

---

## Files that change in `python_agent/` (minimal)

- `agent.py` — add `on_checkpoint` field to `AgentCallbacks`
- `processor.py` — call `await agent.callbacks.on_checkpoint()` at two points
- `tools/registry.py` — add `profile` parameter to `get_default_tools`
- `api_client.py` — Bedrock retry logic (Phase 2.2)
- `tests/test_agent.py`, `tests/test_processor.py` — cover the new callback

Everything else (`context.py`, `messages.py`, `tools/*.py` except registry, etc.) untouched.

## Files that are new

```
python_agent_web/
├── __init__.py
├── app.py                        # FastAPI app, lifespan, router mounts
├── auth.py                       # JWT middleware (Phase 2.1)
├── db.py                         # psycopg pool, query helpers
├── session.py                    # SessionStore, rehydration
├── callbacks.py                  # WebSessionCallbacks → SSE queue
├── rate_limit.py                 # per-user quotas (Phase 2.3)
├── audit.py                      # audit logging (Phase 3.6)
├── observability.py              # OTel init (Phase 2.5)
├── jobs/
│   └── archive.py                # retention cron (Phase 3.4)
├── routes/
│   ├── chat.py                   # POST /chat/message, /chat/stream
│   ├── sessions.py               # POST/GET/DELETE /sessions/{id}
│   └── health.py                 # GET /health
└── schemas.py                    # Pydantic request/response models

Dockerfile                        # multi-stage (Phase 1.7)
docker-compose.yml                # local dev (Phase 1.8)
.dockerignore
_ci/
├── migrations/001_initial.sql    # Phase 1.2
├── migrations/002_rate_limits.sql# Phase 2.3
├── migrations/003_audit_log.sql  # Phase 3.6
├── alembic/                      # Phase 3.1
├── gunicorn.conf.py              # Phase 2.7
├── run.sh                        # entrypoint
└── load/
    └── load_test.py              # Phase 3.2

tests/web/                        # mirrors python_agent_web/
└── conftest.py                   # testcontainers-postgres fixture
```

---

## Things deliberately out of scope for all phases

- **WebSocket** — not needed, tools are safe
- **LangGraph** — not needed, we own the loop
- **Per-node checkpointing** — turn-level is enough
- **Custom binary serialization** — JSONB is already binary, TOAST compresses
- **Sandboxing** — not needed for the customer profile (no `run_bash` / `run_python`)
- **Multi-tenancy beyond `user_id`** — single-tenant (your org), many users

---

## Open questions to resolve before Phase 2

1. **Auth**: which service issues JWTs? If none, how do we authenticate in pre-production?
2. **Retention**: exact day counts for active → archive → delete
3. **Observability backend**: Grafana / Datadog / your org's OTel collector
4. **Tool surface**: concrete list for the customer profile (blocks Phase 4 planning, not Phase 1)
5. **RAG backend**: pgvector / Pinecone / OpenSearch (blocks Phase 4.1, not earlier phases)

These don't block Phase 1. Phase 2.1 (JWT) needs #1 answered.

---

## Effort estimate

| Phase | Solo | Team of 2 | Team of 3 |
|-------|------|-----------|-----------|
| 1 — Foundation | ~1 week | ~3 days | ~2 days |
| 2 — Hardening | ~2 weeks | ~1 week | ~4 days |
| 3 — Scale & ops | ~2 weeks | ~1 week | ~4 days |
| 4 — Domain features | ongoing | ongoing | ongoing |
| **Platform ready** | **~5 weeks** | **~2.5 weeks** | **~2 weeks** |

Phase 4 is product development on top of the platform — duration depends on feature list.

---

## How to use this plan

1. Approve the phase breakdown
2. Start Phase 1.1; each step has explicit files and exit criteria
3. At Phase 1.9 exit, demo end-to-end, get feedback, adjust
4. Move to Phase 2 with the open questions answered
5. Phase 3 kicks in when there's a staging environment and real load
6. Phase 4 runs in parallel with normal product development

Each step is small enough to ship in one focused session.
