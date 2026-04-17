"""Tests for query_databricks."""

import httpx

from thesaurus.tools.query_databricks import QueryDatabricksTool
from thesaurus.tools.registry import get_default_tools
from thesaurus.tools.base import ToolName

_HOST = "https://test.cloud.databricks.com"
_KEY = "dapi-test-token"
_WAREHOUSE = "abc123"


def _tool() -> QueryDatabricksTool:
    return QueryDatabricksTool(host=_HOST, api_key=_KEY, warehouse_id=_WAREHOUSE)


def _response(
    *,
    status: int = 200,
    state: str = "SUCCEEDED",
    columns: list[str] | None = None,
    rows: list[list] | None = None,
    error_message: str | None = None,
    body_override: str | None = None,
) -> httpx.MockTransport:
    """Build a mock transport returning a Statement Execution API response."""
    payload: dict = {"status": {"state": state}}
    if error_message is not None:
        payload["status"]["error"] = {"message": error_message}
    if columns is not None:
        payload["manifest"] = {
            "schema": {"columns": [{"name": c, "type_name": "STRING"} for c in columns]}
        }
    if rows is not None:
        payload["result"] = {"data_array": rows}

    def handler(request: httpx.Request) -> httpx.Response:
        if body_override is not None:
            return httpx.Response(status, content=body_override.encode())
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def _patch_client(monkeypatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


class TestQueryDatabricks:
    async def test_happy_path_select(self, monkeypatch) -> None:
        transport = _response(
            columns=["id", "name"],
            rows=[["1", "Alice"], ["2", "Bob"]],
        )
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="SELECT * FROM t")
        assert "id | name" in result
        assert "-- | ----" in result
        assert "1 | Alice" in result
        assert "2 | Bob" in result
        assert "(2 rows)" in result

    async def test_row_limit_hit(self, monkeypatch) -> None:
        rows = [[str(i), f"row{i}"] for i in range(500)]
        transport = _response(columns=["id", "label"], rows=rows)
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="SELECT * FROM big_table")
        assert "(500 rows — limit reached, add LIMIT or narrow the query)" in result

    async def test_zero_column_response(self, monkeypatch) -> None:
        transport = _response(columns=[], rows=[])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="USE CATALOG foo")
        assert result == "Query executed successfully (no results)."

    async def test_non_succeeded_state(self, monkeypatch) -> None:
        transport = _response(state="RUNNING")
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="SELECT * FROM slow_table")
        assert "Query failed: RUNNING" in result

    async def test_http_error_status(self, monkeypatch) -> None:
        transport = _response(status=401, body_override="unauthorized")
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="SELECT 1")
        assert "HTTP 401" in result
        assert "unauthorized" in result

    async def test_timeout(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("slow")

        _patch_client(monkeypatch, httpx.MockTransport(handler))
        result = await _tool().execute(query="SELECT * FROM huge")
        assert "timed out after 55s" in result

    async def test_network_error(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _patch_client(monkeypatch, httpx.MockTransport(handler))
        result = await _tool().execute(query="SELECT 1")
        assert "Error:" in result
        assert "connection refused" in result

    def test_registry_gating_absent(self) -> None:
        tools = get_default_tools()
        names = {t.name for t in tools}
        assert ToolName.QUERY_DATABRICKS not in names

    def test_registry_gating_present(self) -> None:
        tools = get_default_tools(
            databricks_host=_HOST,
            databricks_api_key=_KEY,
            databricks_warehouse_id=_WAREHOUSE,
        )
        names = {t.name for t in tools}
        assert ToolName.QUERY_DATABRICKS in names

    def test_host_trailing_slash_stripped(self) -> None:
        tool = QueryDatabricksTool(
            host="https://test.cloud.databricks.com/",
            api_key=_KEY,
            warehouse_id=_WAREHOUSE,
        )
        assert tool._host == "https://test.cloud.databricks.com"
