"""Tests for search_confluence."""

import json

import httpx
import pytest

from thesaurus.tools.search_confluence import SearchConfluenceTool

_URL = "https://test.atlassian.net"
_EMAIL = "user@test.com"
_KEY = "test-token"


def _tool() -> SearchConfluenceTool:
    return SearchConfluenceTool(url=_URL, email=_EMAIL, api_key=_KEY)


def _confluence_response(
    pages: list[dict] | None = None,
    status: int = 200,
) -> httpx.MockTransport:
    """Build a mock transport returning a Confluence search response."""
    body = {"results": pages or []}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def _page(
    title: str = "Test Page",
    page_id: str = "123",
    html: str = "<p>hello</p>",
) -> dict:
    return {
        "id": page_id,
        "title": title,
        "body": {"view": {"value": html}},
        "_links": {"webui": f"/spaces/DEV/pages/{page_id}/{title.replace(' ', '+')}"},
    }


def _patch_client(monkeypatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


class TestSearchConfluence:
    async def test_returns_page_content_as_markdown(self, monkeypatch) -> None:
        transport = _confluence_response([_page(
            title="Deploy Guide",
            html="<h1>Steps</h1><p>Run deploy.sh</p>",
        )])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="deploy")
        assert "Deploy Guide" in result
        assert "Steps" in result
        assert "Run deploy.sh" in result

    async def test_multiple_pages(self, monkeypatch) -> None:
        transport = _confluence_response([
            _page(title="Page A", html="<p>alpha</p>"),
            _page(title="Page B", html="<p>beta</p>"),
        ])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="test")
        assert "Page A" in result
        assert "Page B" in result
        assert "alpha" in result
        assert "beta" in result
        assert "---" in result  # separator between pages

    async def test_no_results(self, monkeypatch) -> None:
        transport = _confluence_response([])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="nonexistent")
        assert "No Confluence pages found" in result

    async def test_http_error_status(self, monkeypatch) -> None:
        transport = _confluence_response(status=401)
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="secret")
        assert "HTTP 401" in result

    async def test_timeout(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("slow")

        _patch_client(monkeypatch, httpx.MockTransport(handler))
        result = await _tool().execute(query="slow")
        assert "timed out" in result

    async def test_http_error(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.HTTPError("network error")

        _patch_client(monkeypatch, httpx.MockTransport(handler))
        result = await _tool().execute(query="fail")
        assert "Error:" in result

    async def test_empty_body(self, monkeypatch) -> None:
        transport = _confluence_response([_page(html="")])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="empty")
        assert "(empty page)" in result

    async def test_max_results_clamped(self, monkeypatch) -> None:
        transport = _confluence_response([_page()])
        _patch_client(monkeypatch, transport)
        # Should not crash with out-of-range values
        result = await _tool().execute(query="test", max_results=0)
        assert "Test Page" in result
        result = await _tool().execute(query="test", max_results=100)
        assert "Test Page" in result

    async def test_page_url_includes_domain(self, monkeypatch) -> None:
        transport = _confluence_response([_page(title="Ops", page_id="42")])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="ops")
        assert f"{_URL}/spaces/DEV/pages/42/Ops" in result

    async def test_long_page_truncated(self, monkeypatch) -> None:
        long_html = "<p>" + "x" * 20_000 + "</p>"
        transport = _confluence_response([_page(html=long_html)])
        _patch_client(monkeypatch, transport)
        result = await _tool().execute(query="big")
        assert "... (truncated)" in result

    async def test_cql_escapes_quotes(self, monkeypatch) -> None:
        """Verify double quotes in query are escaped for CQL."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"results": []})

        _patch_client(monkeypatch, httpx.MockTransport(handler))
        await _tool().execute(query='test "quoted" value')
        assert captured
        cql = str(captured[0].url)
        assert '\\"' in cql or "%5C%22" in cql or "%22" not in cql.split("cql=")[1].split("&")[0].replace("%5C%22", "")

    async def test_url_trailing_slash_stripped(self) -> None:
        tool = SearchConfluenceTool(
            url="https://test.atlassian.net/",
            email=_EMAIL,
            api_key=_KEY,
        )
        assert tool._url == "https://test.atlassian.net"
