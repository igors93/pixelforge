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
    GeometricOrbitParameters,
)
from pixel_forge.generators.geometric_orbit.shape_grammar import (
    COMPOSITION_STYLES,
    SHAPE_GRAMMARS,
)


def build_recipe(random_source: Generator) -> GeometricOrbitParameters:
    """Build a complete composition with one coherent shape language."""

    composition_style = COMPOSITION_STYLES[
        int(random_source.integers(0, len(COMPOSITION_STYLES)))
    ]
    grammar = SHAPE_GRAMMARS[composition_style]
    palette = PALETTES[int(random_source.integers(0, len(PALETTES)))]
    background_style = grammar.background_choices[
        int(random_source.integers(0, len(grammar.background_choices)))
    ]

    center_radius = random_source.uniform(0.155, 0.195)
    if composition_style == "symbolic-minimal":
        center_radius = random_source.uniform(0.180, 0.215)
    elif composition_style in {
        "double-square-crown",
        "triangle-explosion",
    }:
        center_radius = random_source.uniform(0.145, 0.175)
    elif composition_style in {"bubble-field", "scattered-constellation"}:
        center_radius = random_source.uniform(0.150, 0.185)

    panels = build_panels(
        random_source,
        style=background_style,
        palette=palette,
        grammar=grammar,
    )
    outer_shapes = build_outer_shapes(
        random_source,
        grammar=grammar,
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
        grammar=grammar,
        composition_style=composition_style,
        background_style=background_style,
        panels=panels,
        outer_shapes=outer_shapes,
        inner_shapes=inner_shapes,
        center_radius=center_radius,
        center_outline_width=random_source.uniform(0.006, 0.010),
        center_shadow_opacity=int(random_source.integers(82, 128)),
        center_highlight_strength=int(random_source.integers(15, 40)),
    )
