"""Composition rules for coherent geometric-orbit layouts."""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from pixel_forge.generators.geometric_orbit.parameters import (
    PaletteDefinition,
    ShapeGrammar,
    ShapeKind,
    ShapeSpec,
)


def _choose_kind(random_source: Generator, grammar: ShapeGrammar) -> ShapeKind:
    value = float(random_source.random())
    if grammar.tertiary_shape is not None and value < grammar.tertiary_probability:
        return grammar.tertiary_shape
    if value < grammar.tertiary_probability + grammar.secondary_probability:
        return grammar.secondary_shape
    return grammar.primary_shape


def _size_multiplier(random_source: Generator, grammar: ShapeGrammar) -> float:
    value = float(random_source.random())
    if value < grammar.large_shape_probability:
        return random_source.uniform(1.45, 2.20)
    if grammar.size_distribution == "wide":
        return random_source.uniform(0.60, 1.65)
    if grammar.size_distribution == "multi-scale":
        if value < 0.55:
            return random_source.uniform(0.55, 0.85)
        return random_source.uniform(0.88, 1.35)
    return random_source.uniform(0.88, 1.16)


def _shape_dimensions(
    random_source: Generator,
    *,
    kind: ShapeKind,
    base_size: float,
    grammar: ShapeGrammar,
) -> tuple[float, float]:
    scale = _size_multiplier(random_source, grammar)
    size = base_size * scale

    if kind in {"circle", "ring"}:
        diameter = size * random_source.uniform(0.88, 1.12)
        return diameter, diameter
    if kind == "square":
        side = size * random_source.uniform(0.92, 1.16)
        return side, side
    if kind == "diamond":
        side = size * random_source.uniform(0.95, 1.25)
        return side, side
    if kind == "triangle":
        return (
            size * random_source.uniform(1.15, 1.90),
            size * random_source.uniform(1.20, 2.00),
        )
    if kind == "capsule":
        return (
            size * random_source.uniform(1.75, 2.65),
            size * random_source.uniform(0.48, 0.78),
        )
    return (
        size * random_source.uniform(1.55, 2.35),
        max(size * random_source.uniform(0.16, 0.25), 0.006),
    )


def _orbit_radius(
    random_source: Generator,
    *,
    grammar: ShapeGrammar,
    center_radius: float,
    index: int,
) -> float:
    ring = index % max(grammar.ring_count, 1)
    base_gap = {
        "single-ring": 0.060,
        "double-ring": 0.045,
        "radial-burst": 0.045,
        "scattered-orbit": 0.060,
        "constellation": 0.075,
    }[grammar.distribution]
    ring_gap = 0.075 * ring
    radius = center_radius + base_gap + ring_gap
    return radius + random_source.uniform(
        -grammar.radial_jitter,
        grammar.radial_jitter,
    )


def _position(
    random_source: Generator,
    *,
    grammar: ShapeGrammar,
    center_radius: float,
    index: int,
    count: int,
) -> tuple[float, float, float]:
    base_angle = math.tau * index / max(count, 1)
    angle = base_angle + random_source.uniform(
        -grammar.angular_jitter,
        grammar.angular_jitter,
    )
    radius = _orbit_radius(
        random_source,
        grammar=grammar,
        center_radius=center_radius,
        index=index,
    )

    if grammar.distribution == "constellation":
        cluster_wave = 0.06 * math.sin(base_angle * 3.0)
        radius += cluster_wave
    elif grammar.distribution == "radial-burst":
        radius += 0.04 * abs(math.sin(base_angle * 2.5))

    return (
        0.5 + radius * math.cos(angle),
        0.5 + radius * math.sin(angle),
        angle,
    )


def _clamp_position(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    padding = 0.025
    half_extent = max(width, height) * 0.72
    low = padding + half_extent
    high = 1.0 - padding - half_extent
    return (
        float(np.clip(center_x, low, high)),
        float(np.clip(center_y, low, high)),
    )


def build_outer_shapes(
    random_source: Generator,
    *,
    grammar: ShapeGrammar,
    palette: PaletteDefinition,
    center_radius: float,
) -> tuple[ShapeSpec, ...]:
    """Build orbiting forms from one dominant shape grammar."""

    count = int(random_source.integers(grammar.min_count, grammar.max_count + 1))
    base_size = {
        "symbolic-minimal": 0.036,
        "double-square-crown": 0.036,
        "triangle-explosion": 0.044,
        "bubble-field": 0.035,
        "scattered-constellation": 0.032,
        "geometric-sun": 0.034,
        "radial-capsules": 0.034,
        "neon-orbit": 0.038,
    }[grammar.name]

    shapes: list[ShapeSpec] = []
    for index in range(count):
        kind = _choose_kind(random_source, grammar)
        width, height = _shape_dimensions(
            random_source,
            kind=kind,
            base_size=base_size,
            grammar=grammar,
        )
        center_x, center_y, angle = _position(
            random_source,
            grammar=grammar,
            center_radius=center_radius,
            index=index,
            count=count,
        )
        center_x, center_y = _clamp_position(
            center_x,
            center_y,
            width,
            height,
        )

        radial_degrees = math.degrees(angle)
        if grammar.distribution == "radial-burst":
            rotation = radial_degrees + random_source.uniform(-24.0, 24.0)
        elif grammar.name in {"radial-capsules", "geometric-sun"}:
            rotation = radial_degrees + 90.0 + random_source.uniform(-8.0, 8.0)
        else:
            rotation = radial_degrees + random_source.uniform(-35.0, 35.0)

        fill = palette.orbit_fill
        if index % 5 == 0 or kind in {"diamond", "triangle"}:
            fill = palette.orbit_secondary

        fill_opacity = 255
        if grammar.allow_outline_only and kind in {"ring", "line"}:
            fill_opacity = 0
        elif grammar.allow_outline_only and random_source.random() < 0.08:
            fill_opacity = 0

        shapes.append(
            ShapeSpec(
                kind=kind,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                rotation_degrees=rotation,
                fill=fill,
                outline=palette.orbit_outline,
                outline_width=random_source.uniform(0.0036, 0.0068),
                shadow_opacity=int(random_source.integers(18, 58)),
                fill_opacity=fill_opacity,
            )
        )

    return tuple(shapes)


_INNER_LAYOUTS: tuple[
    tuple[tuple[float, float, ShapeKind, float, float, float], ...],
    ...,
] = (
    (
        (-0.24, -0.18, "square", 0.18, 0.18, 0.0),
        (0.20, -0.08, "circle", 0.10, 0.10, 0.0),
        (0.16, 0.22, "square", 0.23, 0.23, 0.0),
        (-0.03, 0.10, "line", 0.29, 0.035, -43.0),
        (-0.13, 0.23, "square", 0.07, 0.07, 0.0),
    ),
    (
        (-0.20, 0.16, "square", 0.21, 0.21, 0.0),
        (0.04, 0.16, "square", 0.21, 0.21, 0.0),
        (0.25, 0.06, "circle", 0.12, 0.12, 0.0),
        (0.02, -0.19, "line", 0.30, 0.035, -44.0),
        (0.28, -0.18, "ring", 0.07, 0.07, 0.0),
    ),
    (
        (-0.22, -0.18, "circle", 0.20, 0.20, 0.0),
        (0.15, 0.05, "square", 0.25, 0.25, 0.0),
        (-0.05, 0.20, "diamond", 0.08, 0.08, 45.0),
        (0.24, -0.20, "circle", 0.07, 0.07, 0.0),
        (-0.02, 0.02, "line", 0.28, 0.035, -48.0),
    ),
    (
        (-0.23, -0.20, "ring", 0.17, 0.17, 0.0),
        (0.21, -0.18, "square", 0.12, 0.12, 0.0),
        (0.18, 0.18, "circle", 0.18, 0.18, 0.0),
        (-0.08, 0.20, "square", 0.07, 0.07, 0.0),
        (0.00, 0.03, "line", 0.31, 0.035, -46.0),
    ),
)


def build_inner_shapes(
    random_source: Generator,
    *,
    palette: PaletteDefinition,
    center_radius: float,
) -> tuple[ShapeSpec, ...]:
    """Build a compact symbolic arrangement inside the center disk."""

    layout = _INNER_LAYOUTS[int(random_source.integers(0, len(_INNER_LAYOUTS)))]
    shapes: list[ShapeSpec] = []

    for index, (ox, oy, kind, width_scale, height_scale, rotation) in enumerate(
        layout
    ):
        fill = palette.inner_fill if index % 3 else palette.inner_accent
        fill_opacity = 0 if kind == "ring" else 255
        shapes.append(
            ShapeSpec(
                kind=kind,
                center_x=0.5 + ox * center_radius,
                center_y=0.5 + oy * center_radius,
                width=max(width_scale * center_radius, 0.008),
                height=max(height_scale * center_radius, 0.008),
                rotation_degrees=rotation,
                fill=fill,
                outline=palette.inner_outline,
                outline_width=0.0034,
                shadow_opacity=34,
                fill_opacity=fill_opacity,
            )
        )

    return tuple(shapes)
