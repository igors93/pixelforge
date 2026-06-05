"""Tests for named palette registry and weighted sampling."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_forge.aesthetics.palettes.default_palettes import DEFAULT_PALETTES
from pixel_forge.aesthetics.palettes.palette_registry import (
    build_default_palette_registry,
)
from pixel_forge.core.exceptions import PaletteNotFoundError


def test_default_registry_contains_required_palettes() -> None:
    registry = build_default_palette_registry()
    required = {
        "ocean-depth",
        "solar-flare",
        "midnight-neon",
        "forest-mist",
        "rose-gold",
        "polar-aurora",
        "monochrome-ink",
        "eclipse",
        "iridescent",
        "ultraviolet",
    }
    assert required <= set(registry.names())


def test_get_unknown_palette_raises() -> None:
    registry = build_default_palette_registry()
    with pytest.raises(PaletteNotFoundError):
        registry.get("nonexistent-palette")


def test_sample_returns_a_known_palette() -> None:
    registry = build_default_palette_registry()
    rng = np.random.default_rng(42)
    palette = registry.sample(rng)
    assert palette.name in registry.names()


def test_sample_is_deterministic() -> None:
    registry = build_default_palette_registry()
    p1 = registry.sample(np.random.default_rng(7))
    p2 = registry.sample(np.random.default_rng(7))
    assert p1.name == p2.name


def test_all_default_palettes_have_valid_probabilities() -> None:
    for palette in DEFAULT_PALETTES:
        assert palette.sampling_probability > 0.0


def test_all_default_palettes_have_valid_saturation_range() -> None:
    for palette in DEFAULT_PALETTES:
        assert 0.0 <= palette.min_saturation <= palette.max_saturation <= 1.0


def test_palette_registry_get_returns_correct_palette() -> None:
    registry = build_default_palette_registry()
    palette = registry.get("ocean-depth")
    assert palette.name == "ocean-depth"
