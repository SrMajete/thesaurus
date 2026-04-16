"""Entry point for ``python -m python_agent``."""

from .cli_tui import run
from .config import get_settings


def main() -> None:
    run(get_settings())


if __name__ == "__main__":
    main()
