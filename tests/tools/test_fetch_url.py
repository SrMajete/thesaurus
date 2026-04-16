"""Tests for fetch_url."""

import httpx
import pytest

from python_agent.tools.fetch_url import (
    FetchUrlTool,
    _html_to_markdown,
    _html_to_text,
    _is_html,
    _strip_noisy_elements,
)


class TestPureHelpers:
    def test_is_html_text_html(self) -> None:
        assert _is_html("text/html") is True
        assert _is_html("text/html; charset=utf-8") is True

    def test_is_html_xhtml(self) -> None:
        assert _is_html("application/xhtml+xml") is True

    def test_is_html_false(self) -> None:
        assert _is_html("application/json") is False
        assert _is_html("") is False

    def test_strip_script(self) -> None:
        html = "<html><script>alert(1)</script><body>ok</body></html>"
        cleaned = _strip_noisy_elements(html)
        assert "alert" not in cleaned
        assert "ok" in cleaned

    def test_strip_style(self) -> None:
        html = "<style>body{}</style><p>hi</p>"
        cleaned = _strip_noisy_elements(html)
        assert "body{}" not in cleaned
        assert "hi" in cleaned

    def test_strip_noscript(self) -> None:
        html = "<noscript>no js</noscript><p>hi</p>"
        cleaned = _strip_noisy_elements(html)
        assert "no js" not in cleaned

    def test_html_to_markdown(self) -> None:
        md = _html_to_markdown(
            "<h1>Title</h1><script>x</script><p>hi <b>world</b></p>"
        )
        assert "Title" in md
        assert "**world**" in md
        assert "x" not in md  # script stripped

    def test_html_to_text(self) -> None:
        txt = _html_to_text("<script>x</script><p>hello</p>")
        assert "hello" in txt
        assert "x" not in txt


def _mock_transport(
    *, status: int = 200, content: bytes = b"<p>hi</p>",
    content_type: str = "text/html",
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=content,
            headers={"content-type": content_type},
        )
    return httpx.MockTransport(handler)


class TestFetchUrl:
    async def test_invalid_url_scheme(self) -> None:
        result = await FetchUrlTool().execute(url="ftp://example.com")
        assert "must start with http" in result

    async def test_markdown_format_html(self, monkeypatch) -> None:
        transport = _mock_transport(
            content=b"<h1>Hi</h1><p>world</p>",
            content_type="text/html",
        )
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com")
        assert "Hi" in result
        assert "world" in result

    async def test_text_format(self, monkeypatch) -> None:
        transport = _mock_transport(
            content=b"<p>plain</p>", content_type="text/html",
        )
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(
            url="https://example.com", format="text",
        )
        assert "plain" in result

    async def test_html_format_returns_raw(self, monkeypatch) -> None:
        transport = _mock_transport(
            content=b"<p>raw</p>", content_type="text/html",
        )
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(
            url="https://example.com", format="html",
        )
        assert "<p>raw</p>" in result

    async def test_non_html_returns_raw(self, monkeypatch) -> None:
        transport = _mock_transport(
            content=b'{"ok": true}', content_type="application/json",
        )
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(
            url="https://example.com/data.json",
        )
        assert '"ok": true' in result

    async def test_http_error_status(self, monkeypatch) -> None:
        transport = _mock_transport(status=404)
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com/404")
        assert "HTTP 404" in result

    async def test_response_too_large(self, monkeypatch) -> None:
        huge = b"x" * (6 * 1024 * 1024)
        transport = _mock_transport(content=huge, content_type="text/plain")
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com/big")
        assert "exceeds 5 MB" in result

    async def test_timeout(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("slow")

        transport = httpx.MockTransport(handler)
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(
            url="https://example.com", timeout=1,
        )
        assert "timed out" in result

    async def test_connect_error(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        transport = httpx.MockTransport(handler)
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com")
        assert "could not connect" in result

    async def test_too_many_redirects(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TooManyRedirects("loop")

        transport = httpx.MockTransport(handler)
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com")
        assert "too many redirects" in result

    async def test_generic_http_error(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.HTTPError("boom")

        transport = httpx.MockTransport(handler)
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(url="https://example.com")
        assert "Error: boom" in result

    async def test_default_accept_header_for_unknown_format(
        self, monkeypatch,
    ) -> None:
        """Unknown format values fall back to markdown's Accept header."""
        transport = _mock_transport(
            content=b"<p>x</p>", content_type="text/html",
        )
        self._patch_client(monkeypatch, transport)
        result = await FetchUrlTool().execute(
            url="https://example.com", format="weird",
        )
        # Should still succeed and format as markdown (fallback)
        assert "x" in result

    async def test_timeout_clamping_below_one(self, monkeypatch) -> None:
        transport = _mock_transport()
        self._patch_client(monkeypatch, transport)
        # timeout=0 should clamp to 1
        result = await FetchUrlTool().execute(
            url="https://example.com", timeout=0,
        )
        assert "Fetched" in result

    async def test_timeout_clamping_above_max(self, monkeypatch) -> None:
        transport = _mock_transport()
        self._patch_client(monkeypatch, transport)
        # timeout=999 should clamp to MAX_TIMEOUT (120)
        result = await FetchUrlTool().execute(
            url="https://example.com", timeout=999,
        )
        assert "Fetched" in result

    @staticmethod
    def _patch_client(monkeypatch, transport: httpx.MockTransport) -> None:
        """Replace AsyncClient's default transport with a mock."""
        real_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = transport
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
