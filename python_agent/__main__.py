"""Entry point for ``python -m python_agent``.

Picks the CLI implementation at startup based on ``settings.cli_style``
and dispatches to its ``repl()`` / ``run()``. Imports are lazy so the
unused CLI (and its dependencies) is never loaded. The ``Settings``
instance is constructed once here and passed down — each CLI receives
its config explicitly rather than re-reading the environment.
"""

import asyncio

from .config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.cli_style == "tui":
        # Textual owns its own event loop; don't wrap with asyncio.run.
        from .cli_tui import run
        run(settings)
        return
    if settings.cli_style == "rich":
        from .cli_rich import repl
    else:
        from .cli import repl
    asyncio.run(repl(settings))


if __name__ == "__main__":
    main()
