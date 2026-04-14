"""Entry point for ``python -m python_agent``.

Picks the CLI implementation at startup based on ``settings.cli_style``
and dispatches to its ``repl()``. Imports are lazy so the unused CLI
(and its dependencies) is never loaded.
"""

import asyncio

from .config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.cli_style == "rich":
        from .cli_rich import repl
    else:
        from .cli import repl
    asyncio.run(repl())


if __name__ == "__main__":
    main()
