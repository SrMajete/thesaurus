"""Entry point for ``python -m thesaurus``."""

from .adapters.config import get_settings
from .adapters.tui import run


def main() -> None:
    run(get_settings())


if __name__ == "__main__":
    main()
