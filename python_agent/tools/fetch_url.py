"""Fetch URL tool — retrieve web pages as readable content.

Performs an async HTTP GET and converts HTML to markdown (default),
plain text, or returns raw HTML.  Non-HTML responses are returned as-is.
"""

import re

import httpx
from markdownify import markdownify
from typing import Any

from .base import ToolName

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_OUTPUT_CHARS = 50_000
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)

_ACCEPT_HEADERS: dict[str, str] = {
    "markdown": (
        "text/markdown;q=1.0, text/html;q=0.9, "
        "text/plain;q=0.8, */*;q=0.1"
    ),
    "text": (
        "text/plain;q=1.0, text/html;q=0.8, */*;q=0.1"
    ),
    "html": (
        "text/html;q=1.0, application/xhtml+xml;q=0.9, */*;q=0.1"
    ),
}


def _is_html(content_type: str) -> bool:
    """Check if the Content-Type header indicates HTML."""
    mime = content_type.split(";")[0].strip().lower()
    return mime in ("text/html", "application/xhtml+xml")


_STRIP_TAGS_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_noisy_elements(html: str) -> str:
    """Remove script, style, and noscript elements and their content."""
    return _STRIP_TAGS_RE.sub("", html)


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown, stripping scripts and styles."""
    return markdownify(_strip_noisy_elements(html))


def _html_to_text(html: str) -> str:
    """Strip all HTML tags, returning plain text."""
    # markdownify with no conversion produces plain text with whitespace cleanup.
    return markdownify(_strip_noisy_elements(html), convert=[""]).strip()


def _truncate(text: str) -> str:
    """Truncate to the output character limit."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n\n... truncated ({len(text)} chars total)"


class FetchUrlTool:
    """Fetch a URL and return its content as markdown, text, or raw HTML."""

    name = ToolName.FETCH_URL
    description = (
        "Fetch a web page and return its content. Converts HTML to markdown "
        "by default. Use for reading documentation, API references, GitHub "
        "issues, or any public web content. Do not use for authenticated or "
        "private URLs."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must start with http:// or https://).",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "text", "html"],
                "description": (
                    "Output format. 'markdown' (default) converts HTML to "
                    "markdown. 'text' strips all tags. 'html' returns raw HTML."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": f"Request timeout in seconds (default {_DEFAULT_TIMEOUT}, max {_MAX_TIMEOUT}).",
            },
        },
        "required": ["url"],
    }
    needs_permission = True
    is_parallelizable = True
    is_intercepted = False

    async def execute(
        self, *, url: str, format: str = "markdown", timeout: int = _DEFAULT_TIMEOUT,
    ) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"

        timeout = max(1, min(timeout, _MAX_TIMEOUT))
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": _ACCEPT_HEADERS.get(format, _ACCEPT_HEADERS["markdown"]),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, max_redirects=20,
            ) as client:
                response = await client.get(
                    url, headers=headers, timeout=timeout,
                )
        except httpx.TimeoutException:
            return f"Error: request timed out after {timeout}s for {url}"
        except httpx.ConnectError as e:
            return f"Error: could not connect to {url}: {e}"
        except httpx.TooManyRedirects:
            return f"Error: too many redirects for {url}"
        except httpx.HTTPError as e:
            return f"Error: {e}"

        if response.status_code >= 400:
            return f"Error: HTTP {response.status_code} for {url}"

        byte_count = len(response.content)
        if byte_count > _MAX_RESPONSE_BYTES:
            return "Error: response exceeds 5 MB limit."

        content_type = response.headers.get("content-type", "")
        text = response.text

        if format == "html" or not _is_html(content_type):
            body = text
        elif format == "text":
            body = _html_to_text(text)
        else:
            body = _html_to_markdown(text)

        body = _truncate(body)
        return f"Fetched {url} ({content_type}, {byte_count} bytes)\n\n{body}"
