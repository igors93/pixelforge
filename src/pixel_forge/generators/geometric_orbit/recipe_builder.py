"""Deterministic recipe builder for geometric-orbit."""

from __future__ import annotations

from numpy.random import Generator

from pixel_forge.generators.geometric_orbit.background import build_panels
from pixel_forge.generators.geometric_orbit.composition import (
    build_inner_shapes,
    build_outer_shapes,
)
from pixel_forge.generators.geometric_orbit.palettes import PALETTES
from pixel_forge.generators.geometric_orbit.parameters import (
    BackgroundStyle,
    CompositionStyle,
    GeometricOrbitParameters,
)

_COMPOSITION_STYLES: tuple[CompositionStyle, ...] = (
    "balanced-orbit",
    "capsule-sun",
    "bubble-ring",
    "shard-burst",
    "double-ring",
    "minimal-symbols",
)

_BACKGROUND_STYLES: tuple[BackgroundStyle, ...] = (
    "diagonal-panels",
    "angular-folds",
    "offset-ribbons",
    "split-field",
)


def build_recipe(random_source: Generator) -> GeometricOrbitParameters:
    """Build a complete high-contrast composition recipe."""

    palette = PALETTES[int(random_source.integers(0, len(PALETTES)))]
    composition_style = _COMPOSITION_STYLES[
        int(random_source.integers(0, len(_COMPOSITION_STYLES)))
    ]
    background_style = _BACKGROUND_STYLES[
        int(random_source.integers(0, len(_BACKGROUND_STYLES)))
    ]

    center_radius = random_source.uniform(0.155, 0.205)
    if composition_style == "minimal-symbols":
        center_radius = random_source.uniform(0.175, 0.215)
    elif composition_style == "double-ring":
        center_radius = random_source.uniform(0.145, 0.180)

    panels = build_panels(
        random_source,
        style=background_style,
        palette=palette,
    )
    outer_shapes = build_outer_shapes(
        random_source,
        style=composition_style,
        palette=palette,
        center_radius=center_radius,
    )
    inner_shapes = build_inner_shapes(
        random_source,
        palette=palette,
        center_radius=center_radius,
    )

    return GeometricOrbitParameters(
        palette=palette,
        composition_style=composition_style,
        background_style=background_style,
        panels=panels,
        outer_shapes=outer_shapes,
        inner_shapes=inner_shapes,
        center_radius=center_radius,
        center_outline_width=random_source.uniform(0.006, 0.010),
        center_shadow_opacity=int(random_source.integers(75, 120)),
        center_highlight_strength=int(random_source.integers(14, 38)),
    )
