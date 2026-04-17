# Databricks Query Tool — Implementation Plan

Add a `query_databricks` tool that executes SQL against a Databricks SQL
warehouse. One tool covers the full investigation workflow: schema discovery,
table exploration, data querying, and analysis — SQL is already the universal
interface for all of these.

---

## Design decisions

### One tool, not multiple

Don't create separate `list_schemas`, `list_tables`, `describe_table`,
`run_query` tools. The agent writes SQL directly:

- `SHOW SCHEMAS` → lists schemas
- `SHOW TABLES IN schema` → lists tables
- `DESCRIBE TABLE x` → shows columns and types
- `SELECT * FROM x LIMIT 10` → previews data
- Any analytical query → answers the user's question

Multiple tools would add token overhead (~200 tokens per schema × every API
call × every turn), increase the model's decision surface, and gain nothing
that SQL doesn't already provide.

### Permission model

`needs_permission = True` — SQL can mutate data (`INSERT`, `DELETE`, `DROP`).
The user sees and approves every query before it hits the warehouse. For a
production data source, this is a feature: the user catches mistakes before
execution, not after.

### Parallelism

`is_parallelizable = True` — HTTP calls are async and independent. The agent
can fire multiple exploration queries in one turn (e.g., `DESCRIBE TABLE x`
+ `DESCRIBE TABLE y` in parallel).

### Analysis workflow

SQL handles data access and aggregation. The existing `run_python` tool
handles computation SQL can't do (statistical tests, charts, complex
transformations). The agent naturally chains these:

| Turn | Tool calls | Purpose |
|------|-----------|---------|
| 1 | `make_plan` + `query_databricks("SHOW SCHEMAS")` | Plan investigation |
| 2 | `query_databricks("SHOW TABLES IN sales")` + `query_databricks("SHOW TABLES IN geo")` | Parallel exploration |
| 3 | `query_databricks("DESCRIBE TABLE sales.orders")` + `query_databricks("DESCRIBE TABLE geo.regions")` | Parallel schema inspection |
| 4 | `query_databricks("SELECT r.name, AVG(o.total) ...")` | The actual query |
| 5 | `run_python` (if needed for analysis) | Post-processing |
| 6 | `send_response` | Answer with context |

### No new dependencies

The Databricks SQL Statement Execution API is one POST endpoint. `httpx`
(already in `requirements.txt`) is sufficient. The `databricks-sdk` (15MB)
would be YAGNI — add it later only if OAuth/OIDC auth, notebook execution,
or job management are needed.

---

## Files to change

### 1. `tools/base.py` — add tool name

At line 47, add to `ToolName`:

```python
QUERY_DATABRICKS = "query_databricks"
```

### 2. `tools/query_databricks.py` — new file

Follows the `search_confluence.py` pattern exactly: optional external service
tool, credentials at init, httpx for HTTP, truncated text output.

```python
"""Query Databricks tool — SQL execution against a Databricks SQL warehouse.

Executes SQL via the Databricks SQL Statement Execution API and returns
results as a formatted text table. Supports schema exploration (SHOW, DESCRIBE)
and data queries (SELECT, aggregations, joins).

Optional tool — only registered when DATABRICKS_HOST, DATABRICKS_TOKEN,
and DATABRICKS_WAREHOUSE_ID are all configured.
"""

from typing import Any

import httpx

from ._helpers import truncate
from .base import ToolName

_TIMEOUT = 55
_ROW_LIMIT = 500


class QueryDatabricksTool:
    """Execute SQL against a Databricks SQL warehouse."""

    name = ToolName.QUERY_DATABRICKS
    description = (
        "Execute a SQL query against Databricks SQL warehouse. Use for "
        "querying data, exploring schemas (SHOW SCHEMAS, SHOW TABLES, "
        "DESCRIBE TABLE), and any SQL operation. Always use LIMIT on "
        "SELECT queries to avoid huge result sets. Returns results as "
        "a formatted table."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute.",
            },
        },
        "required": ["query"],
    }
    needs_permission = True
    is_parallelizable = True
    is_intercepted = False

    def __init__(self, host: str, token: str, warehouse_id: str) -> None:
        self._host = host.rstrip("/")
        self._token = token
        self._warehouse_id = warehouse_id

    async def execute(self, *, query: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._host}/api/2.0/sql/statements",
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={
                        "warehouse_id": self._warehouse_id,
                        "statement": query,
                        "wait_timeout": "50s",
                        "disposition": "INLINE",
                        "format": "JSON_ARRAY",
                        "row_limit": _ROW_LIMIT,
                    },
                    timeout=_TIMEOUT,
                )
        except httpx.TimeoutException:
            return f"Error: Databricks query timed out after {_TIMEOUT}s."
        except httpx.HTTPError as e:
            return f"Error: {e}"

        if resp.status_code >= 400:
            msg = resp.json().get("message", resp.text)
            return f"Error: Databricks returned HTTP {resp.status_code}: {msg}"

        data = resp.json()
        status = data.get("status", {}).get("state")
        if status != "SUCCEEDED":
            error = data.get("status", {}).get("error", {}).get("message", status)
            return f"Query failed: {error}"

        columns = [
            c["name"]
            for c in data.get("manifest", {}).get("schema", {}).get("columns", [])
        ]
        rows = data.get("result", {}).get("data_array", [])

        if not columns:
            return "Query executed successfully (no results)."

        # Build column-aligned text table
        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        def fmt_row(values: list) -> str:
            return " | ".join(
                str(v).ljust(col_widths[i]) for i, v in enumerate(values)
            )

        header = fmt_row(columns)
        separator = "-+-".join("-" * w for w in col_widths)
        formatted = [fmt_row(row) for row in rows]

        row_count = len(rows)
        footer = f"\n({row_count} row{'s' if row_count != 1 else ''})"
        if row_count >= _ROW_LIMIT:
            footer = f"\n({row_count} rows — limit reached, add LIMIT or narrow the query)"

        table = "\n".join([header, separator] + formatted) + footer
        return truncate(table)
```

### 3. `adapters/config.py` — three optional fields

Add alongside the existing Confluence settings:

```python
# Databricks (optional — query tool only registers when all three are set)
databricks_host: str | None = None          # e.g. https://myworkspace.cloud.databricks.com
databricks_token: str | None = None         # PAT token
databricks_warehouse_id: str | None = None  # SQL warehouse ID
```

### 4. `tools/registry.py` — conditional registration

Extend `get_default_tools` signature and add the conditional block:

```python
def get_default_tools(
    *,
    confluence_url: str | None = None,
    confluence_email: str | None = None,
    confluence_api_key: str | None = None,
    databricks_host: str | None = None,
    databricks_token: str | None = None,
    databricks_warehouse_id: str | None = None,
) -> list[Tool]:
    tools: list[Tool] = [
        # ... existing tools unchanged ...
    ]
    if confluence_url and confluence_email and confluence_api_key:
        from .search_confluence import SearchConfluenceTool
        tools.append(SearchConfluenceTool(
            url=confluence_url,
            email=confluence_email,
            api_key=confluence_api_key,
        ))
    if databricks_host and databricks_token and databricks_warehouse_id:
        from .query_databricks import QueryDatabricksTool
        tools.append(QueryDatabricksTool(
            host=databricks_host,
            token=databricks_token,
            warehouse_id=databricks_warehouse_id,
        ))
    return tools
```

### 5. `adapters/tui.py` — thread new params through

At line 752 where `get_default_tools` is called, add the Databricks settings:

```python
tools=get_default_tools(
    confluence_url=settings.confluence_url,
    confluence_email=settings.confluence_email,
    confluence_api_key=settings.confluence_api_key,
    databricks_host=settings.databricks_host,
    databricks_token=settings.databricks_token,
    databricks_warehouse_id=settings.databricks_warehouse_id,
),
```

---

## API details

### Endpoint

```
POST {host}/api/2.0/sql/statements
```

### Request body

```json
{
    "warehouse_id": "abc123def456",
    "statement": "SELECT * FROM sales.orders LIMIT 10",
    "wait_timeout": "50s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY",
    "row_limit": 500
}
```

- `wait_timeout: "50s"` — API maximum; covers most analytical queries on a
  warm warehouse. Queries exceeding this return `RUNNING` status, which the
  tool reports as an error. Polling for long-running queries is a v2 feature.
- `disposition: "INLINE"` — forces results inline in the response JSON.
  Without this, large results may use `EXTERNAL_LINKS` disposition requiring
  separate fetches.
- `row_limit: 500` — caps results server-side. Prevents multi-megabyte
  responses that `truncate()` would have to chop client-side. The tool's
  footer tells the agent when the limit was reached.

### Response shape

```json
{
    "status": {"state": "SUCCEEDED"},
    "manifest": {
        "schema": {
            "columns": [
                {"name": "id", "type_name": "INT"},
                {"name": "region", "type_name": "STRING"}
            ]
        }
    },
    "result": {
        "data_array": [
            ["1", "EMEA"],
            ["2", "APAC"]
        ]
    }
}
```

All values in `data_array` are strings. The tool formats them into a
column-aligned text table with a row count footer.

---

## Configuration

```env
DATABRICKS_HOST=https://myworkspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi0123456789abcdef
DATABRICKS_WAREHOUSE_ID=abc123def456
```

Tool only appears when all three are set — zero impact when unconfigured.
Same pattern as Confluence.

---

## Known limitations (v1)

1. **Long-running queries**: Queries exceeding 50s return a "Query failed:
   RUNNING" error. Fix: add polling loop that checks
   `GET /api/2.0/sql/statements/{statement_id}` until completion. Deferred
   because most interactive queries complete within 50s on a warm warehouse.

2. **Auth**: PAT token only. OAuth, Azure AD, and service principal auth
   require the `databricks-sdk`. Add when needed.

3. **Large results**: Capped at 500 rows server-side + `truncate()` client-side.
   The tool description instructs the model to use LIMIT, and the footer warns
   when the cap is hit. For truly large exports, the user should use Databricks
   directly.

---

## What stays untouched

- `core/agent.py` — protocol-typed, tool-agnostic
- `core/processor.py` — dispatches by name, no tool-specific logic
- `core/ports.py` — no tool knowledge
- `adapters/anthropic_llm.py` — no tool knowledge
- `tools/base.py` — only one line added (the `ToolName` entry)
- All existing tools — no changes
