"""Search Confluence tool — RAG-style page search and content retrieval.

Searches Confluence Cloud via CQL and returns matching pages as
markdown. The search API's ``expand=body.view`` parameter inlines
each page's HTML body, which is converted to markdown via
``markdownify`` (shared dependency with ``fetch_url``).

Optional tool — only registered when ``CONFLUENCE_URL``,
``CONFLUENCE_EMAIL``, and ``CONFLUENCE_API_KEY`` are all configured.
"""

from typing import Any

import httpx
from markdownify import markdownify

from ._helpers import truncate
from .base import ToolName

_TIMEOUT = 30
_PER_PAGE_CHARS = 8_000


class SearchConfluenceTool:
    """Search Confluence pages and return their content as markdown."""

    name = ToolName.SEARCH_CONFLUENCE
    description = (
        "Search Confluence pages by text query. Returns matching page "
        "titles and their full content as markdown. Use for looking up "
        "internal documentation, runbooks, design docs, or any "
        "knowledge stored in Confluence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — matched against page titles and body text.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of pages to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    }
    needs_permission = False
    is_parallelizable = True
    is_intercepted = False

    def __init__(self, url: str, email: str, api_key: str) -> None:
        self._url = url.rstrip("/")
        self._auth = httpx.BasicAuth(email, api_key)

    async def execute(self, *, query: str, max_results: int = 5) -> str:
        max_results = max(1, min(max_results, 10))
        escaped = query.replace('"', '\\"')
        cql = f'text~"{escaped}" OR title~"{escaped}"'

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._url}/wiki/rest/api/content/search",
                    params={
                        "cql": cql,
                        "limit": max_results,
                        "expand": "body.view",
                    },
                    auth=self._auth,
                    timeout=_TIMEOUT,
                )
        except httpx.TimeoutException:
            return f"Error: Confluence search timed out after {_TIMEOUT}s."
        except httpx.HTTPError as e:
            return f"Error: {e}"

        if resp.status_code >= 400:
            return f"Error: Confluence returned HTTP {resp.status_code}."

        results = resp.json().get("results", [])
        if not results:
            return f"No Confluence pages found for: {query}"

        pages: list[str] = []
        for page in results:
            title = page.get("title", "(untitled)")
            page_url = self._url + page.get("_links", {}).get("webui", "")
            html = page.get("body", {}).get("view", {}).get("value", "")
            body = markdownify(html).strip() if html else "(empty page)"
            if len(body) > _PER_PAGE_CHARS:
                body = body[:_PER_PAGE_CHARS] + "\n\n... (truncated)"
            pages.append(f"## {title}\n{page_url}\n\n{body}")

        output = "\n\n---\n\n".join(pages)
        return truncate(output)
