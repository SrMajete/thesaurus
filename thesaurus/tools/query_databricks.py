"""Query Databricks tool — SQL execution against a Databricks SQL warehouse.

Executes SQL via the Statement Execution API (one POST to
``/api/2.0/sql/statements``) and returns results as a pipe-delimited text
table. Covers schema exploration (``SHOW``, ``DESCRIBE``) and data queries
(``SELECT`` and friends) through a single ``query`` parameter.

Optional tool — only registered when ``DATABRICKS_HOST``, ``DATABRICKS_API_KEY``,
and ``DATABRICKS_WAREHOUSE_ID`` are all configured.
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
        "a pipe-delimited text table."
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
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    def __init__(self, host: str, api_key: str, warehouse_id: str) -> None:
        self._host = host.rstrip("/")
        self._api_key = api_key
        self._warehouse_id = warehouse_id

    async def execute(self, *, query: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._host}/api/2.0/sql/statements",
                    headers={"Authorization": f"Bearer {self._api_key}"},
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
                resp.raise_for_status()
        except httpx.TimeoutException:
            return f"Error: Databricks query timed out after {_TIMEOUT}s."
        except httpx.HTTPStatusError as e:
            return (
                f"Error: Databricks returned HTTP {e.response.status_code}: "
                f"{e.response.text[:500]}"
            )
        except httpx.HTTPError as e:
            return f"Error: {e}"

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

        header = " | ".join(columns)
        separator = " | ".join("-" * len(c) for c in columns)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)

        row_count = len(rows)
        if row_count >= _ROW_LIMIT:
            footer = f"\n({row_count} rows — limit reached, add LIMIT or narrow the query)"
        else:
            footer = f"\n({row_count} row{'s' if row_count != 1 else ''})"

        table = "\n".join(part for part in [header, separator, body] if part) + footer
        return truncate(table)
