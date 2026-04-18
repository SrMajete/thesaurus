"""Tests for the TUI startup banner."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from textual.widgets import Static
from textual_image.widget import Image as ImageWidget

from thesaurus.adapters.tui import (
    _BANNER,
    _BANNER_IMAGE_PATH,
    _build_banner,
)


def test_banner_image_ships_with_package() -> None:
    """The composite banner PNG must exist so the TUI can load it at runtime."""
    assert _BANNER_IMAGE_PATH.is_file(), f"missing asset: {_BANNER_IMAGE_PATH}"


def test_banner_without_image_is_wordmark_only() -> None:
    """Terminals that lack a graphics protocol see the figlet wordmark."""
    widget = _build_banner(with_image=False)
    assert isinstance(widget, Static)
    assert _BANNER in str(widget.render())


def test_banner_with_image_is_image_widget() -> None:
    """Graphics-capable terminals see the composite banner PNG."""
    widget = _build_banner(with_image=True)
    assert isinstance(widget, ImageWidget)


@pytest.mark.asyncio
async def test_banner_docks_at_top_of_screen() -> None:
    """Layout regression guard: the banner widget must sit at screen y=0.

    Catches accidental removal of ``dock: top`` or a compose-order
    rearrangement that would place the banner inside ``#messages``
    (where it would scroll out of view).
    """
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
    from thesaurus.adapters import tui as tui_mod
    from thesaurus.adapters.config import Settings

    tui_mod.make_llm_client = lambda s: MagicMock()  # type: ignore[assignment]
    tui_mod.fetch_max_context_tokens = lambda s: 200_000  # type: ignore[assignment]

    app = tui_mod.AgentApp(Settings())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Banner is either a Static (wordmark) or an ImageWidget — both are
        # the first Screen-level child in compose() order.
        banner = next(
            w for w in app.screen.children
            if "banner-wordmark" in w.classes or "banner-image" in w.classes
        )
        assert banner.region.y == 0, (
            f"banner should dock at y=0, got region={banner.region}"
        )
