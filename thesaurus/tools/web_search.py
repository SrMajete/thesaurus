"""Web search tool — DuckDuckGo via the ``ddgs`` library.

Returns the top matching results as a numbered list with title, URL, and
snippet.  The model pairs this with ``fetch_url`` when it needs the full
content of a specific result.
"""

import asyncio
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from ._helpers import truncate
from .base import ToolName

_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20


class WebSearchTool:
    """Search the web and return titles + URLs + snippets."""

    name = ToolName.WEB_SEARCH
    description = (
        "Search the web for up-to-date information. Returns titles, URLs, "
        "and snippets for the top results. Use this when you don't know the "
        "exact URL; follow up with fetch_url to read a specific result in full."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Number of results to return (default "
                    f"{_DEFAULT_MAX_RESULTS}, clamped to [1, {_MAX_RESULTS_CAP}])."
                ),
            },
        },
        "required": ["query"],
    }
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    async def execute(
        self, *, query: str, max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> str:
        if not query or not query.strip():
            return "Error: query must not be empty."

        n = max(1, min(max_results, _MAX_RESULTS_CAP))

        try:
            results = await asyncio.to_thread(_search_sync, query, n)
        except RatelimitException:
            return (
                "Error: DuckDuckGo rate-limited the request. "
                "Try again shortly."
            )
        except TimeoutException:
            return "Error: search request timed out."
        except DDGSException as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

        if not results:
            return f"No results for: {query}"

        lines: list[str] = [f'Search results for: "{query}"', ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip() or "(untitled)"
            url = r.get("href", "").strip()
            snippet = r.get("body", "").strip()
            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return truncate("\n".join(lines).rstrip())


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run the blocking DDGS call. Extracted so the async wrapper has a
    single line to send to ``to_thread``, and so tests can patch it cleanly.
    """
    with DDGS() as client:
        return list(client.text(query, max_results=max_results))
