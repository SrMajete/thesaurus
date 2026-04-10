"""Entry point for ``python -m python_agent``."""

import asyncio

from .cli import repl

asyncio.run(repl())
