"""Tests for the web_search tool."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from thesaurus.tools.web_search import WebSearchTool


def _result(title: str = "A Title", href: str = "https://example.com",
            body: str = "A snippet.") -> dict[str, Any]:
    return {"title": title, "href": href, "body": body}


class TestInputValidation:
    async def test_empty_query(self) -> None:
        result = await WebSearchTool().execute(query="")
        assert "must not be empty" in result

    async def test_whitespace_query(self) -> None:
        result = await WebSearchTool().execute(query="   \t\n  ")
        assert "must not be empty" in result

    async def test_max_results_clamped_low(self, monkeypatch) -> None:
        captured: dict[str, int] = {}

        def fake_search(query: str, max_results: int) -> list:
            captured["n"] = max_results
            return []

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", fake_search,
        )
        await WebSearchTool().execute(query="x", max_results=0)
        assert captured["n"] == 1

    async def test_max_results_clamped_high(self, monkeypatch) -> None:
        captured: dict[str, int] = {}

        def fake_search(query: str, max_results: int) -> list:
            captured["n"] = max_results
            return []

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", fake_search,
        )
        await WebSearchTool().execute(query="x", max_results=999)
        assert captured["n"] == 20

    async def test_max_results_negative_clamped(self, monkeypatch) -> None:
        captured: dict[str, int] = {}

        def fake_search(query: str, max_results: int) -> list:
            captured["n"] = max_results
            return []

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", fake_search,
        )
        await WebSearchTool().execute(query="x", max_results=-5)
        assert captured["n"] == 1

    async def test_default_max_results(self, monkeypatch) -> None:
        captured: dict[str, int] = {}

        def fake_search(query: str, max_results: int) -> list:
            captured["n"] = max_results
            return []

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", fake_search,
        )
        await WebSearchTool().execute(query="x")
        assert captured["n"] == 5


class TestBehavior:
    async def test_successful_formatting(self, monkeypatch) -> None:
        results = [
            _result("Python asyncio", "https://docs.python.org/3/library/asyncio.html",
                    "asyncio is a library..."),
            _result("Real Python", "https://realpython.com/async-io-python/",
                    "Tutorial content here"),
        ]

        def fake_search(query: str, max_results: int) -> list:
            return results

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", fake_search,
        )
        out = await WebSearchTool().execute(query="python async")

        assert 'Search results for: "python async"' in out
        assert "1. Python asyncio" in out
        assert "https://docs.python.org/3/library/asyncio.html" in out
        assert "asyncio is a library..." in out
        assert "2. Real Python" in out
        assert "https://realpython.com/async-io-python/" in out

    async def test_empty_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync",
            lambda q, n: [],
        )
        out = await WebSearchTool().execute(query="asdfghjkl zxcvbnm")
        assert "No results for: asdfghjkl zxcvbnm" in out

    async def test_missing_title_shows_untitled(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync",
            lambda q, n: [_result(title="", href="https://x.com", body="body")],
        )
        out = await WebSearchTool().execute(query="x")
        assert "(untitled)" in out

    async def test_missing_url_or_body_handled(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync",
            lambda q, n: [{"title": "Title only"}],  # no href or body
        )
        out = await WebSearchTool().execute(query="x")
        assert "Title only" in out

    async def test_rate_limit_exception(self, monkeypatch) -> None:
        def raising(query: str, max_results: int) -> list:
            raise RatelimitException("slow down")

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", raising,
        )
        out = await WebSearchTool().execute(query="x")
        assert "rate-limited" in out
        assert "Try again" in out

    async def test_timeout_exception(self, monkeypatch) -> None:
        def raising(query: str, max_results: int) -> list:
            raise TimeoutException("slow")

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", raising,
        )
        out = await WebSearchTool().execute(query="x")
        assert "timed out" in out

    async def test_ddgs_exception(self, monkeypatch) -> None:
        def raising(query: str, max_results: int) -> list:
            raise DDGSException("backend failure")

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", raising,
        )
        out = await WebSearchTool().execute(query="x")
        assert "Error:" in out
        assert "backend failure" in out

    async def test_generic_exception(self, monkeypatch) -> None:
        def raising(query: str, max_results: int) -> list:
            raise RuntimeError("unexpected boom")

        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync", raising,
        )
        out = await WebSearchTool().execute(query="x")
        assert "Error:" in out
        assert "unexpected boom" in out

    async def test_output_truncated(self, monkeypatch) -> None:
        huge = [_result(title=f"t{i}", href=f"https://x.com/{i}", body="Y" * 20_000)
                for i in range(100)]
        monkeypatch.setattr(
            "thesaurus.tools.web_search._search_sync",
            lambda q, n: huge,
        )
        out = await WebSearchTool().execute(query="x", max_results=20)
        assert "truncated" in out


class TestSearchSyncIntegration:
    """The _search_sync helper wraps DDGS() with a context manager."""

    def test_search_sync_uses_ddgs_context_manager(self, monkeypatch) -> None:
        from thesaurus.tools import web_search

        enters: list[bool] = []

        class FakeDDGS:
            def __enter__(self) -> "FakeDDGS":
                enters.append(True)
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def text(self, query: str, max_results: int) -> list:
                return [{"title": "T", "href": "https://x", "body": "B"}]

        monkeypatch.setattr(web_search, "DDGS", FakeDDGS)
        out = web_search._search_sync("q", 3)
        assert out == [{"title": "T", "href": "https://x", "body": "B"}]
        assert enters == [True]


class TestToolProtocol:
    def test_has_required_attributes(self) -> None:
        t = WebSearchTool()
        assert t.name == "web_search"
        assert t.needs_permission is False
        assert t.is_parallelizable is True
        assert t.is_intercepted is False
        assert "query" in t.input_schema["properties"]
        assert t.input_schema["required"] == ["query"]
