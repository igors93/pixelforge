"""Composition rules for clean, balanced orbit layouts."""

from __future__ import annotations

import math

from numpy.random import Generator

from pixel_forge.generators.geometric_orbit.parameters import (
    CompositionStyle,
    PaletteDefinition,
    ShapeKind,
    ShapeSpec,
)


_STYLE_SHAPES: dict[CompositionStyle, tuple[ShapeKind, ...]] = {
    "balanced-orbit": ("circle", "square", "capsule", "triangle"),
    "capsule-sun": ("capsule", "capsule", "capsule", "circle"),
    "bubble-ring": ("circle", "circle", "circle", "square"),
    "shard-burst": ("triangle", "triangle", "capsule", "square"),
    "double-ring": ("circle", "square", "capsule", "triangle"),
    "minimal-symbols": ("square", "circle", "line"),
}


def _clamp_center(value: float, half_extent: float, padding: float = 0.035) -> float:
    return min(max(value, padding + half_extent), 1.0 - padding - half_extent)


def build_outer_shapes(
    random_source: Generator,
    *,
    style: CompositionStyle,
    palette: PaletteDefinition,
    center_radius: float,
) -> tuple[ShapeSpec, ...]:
    """Create orbiting forms with even angular coverage and safe canvas padding."""

    count_ranges: dict[CompositionStyle, tuple[int, int]] = {
        "balanced-orbit": (14, 22),
        "capsule-sun": (16, 26),
        "bubble-ring": (12, 20),
        "shard-burst": (14, 24),
        "double-ring": (18, 28),
        "minimal-symbols": (8, 13),
    }
    count = int(random_source.integers(*count_ranges[style]))
    shape_pool = _STYLE_SHAPES[style]
    ring_count = 2 if style == "double-ring" else 1

    shapes: list[ShapeSpec] = []
    for index in range(count):
        ring = index % ring_count
        base_angle = math.tau * index / count
        angle = base_angle + random_source.uniform(-0.055, 0.055)

        orbit_radius = center_radius * (1.48 + 0.28 * ring)
        orbit_radius += random_source.uniform(-0.018, 0.028)

        kind = shape_pool[int(random_source.integers(0, len(shape_pool)))]
        base_size = random_source.uniform(0.026, 0.058)

        if kind == "circle":
            width = height = base_size * random_source.uniform(0.80, 1.18)
        elif kind == "square":
            width = base_size * random_source.uniform(0.85, 1.25)
            height = width
        elif kind == "triangle":
            width = base_size * random_source.uniform(1.10, 1.75)
            height = base_size * random_source.uniform(1.05, 1.70)
        elif kind == "capsule":
            width = base_size * random_source.uniform(1.70, 2.55)
            height = base_size * random_source.uniform(0.55, 0.90)
        else:
            width = base_size * random_source.uniform(1.50, 2.40)
            height = max(base_size * 0.20, 0.007)

        center_x = 0.5 + orbit_radius * math.cos(angle)
        center_y = 0.5 + orbit_radius * math.sin(angle)
        center_x = _clamp_center(center_x, width / 2.0)
        center_y = _clamp_center(center_y, height / 2.0)

        radial_rotation = math.degrees(angle)
        if style in {"capsule-sun", "shard-burst"}:
            rotation = radial_rotation + random_source.uniform(-18.0, 18.0)
        else:
            rotation = radial_rotation + 90.0 + random_source.uniform(-16.0, 16.0)

        fill = palette.orbit_fill
        if style == "capsule-sun" and index % 3 == 0:
            fill = palette.panel_accent
        if style == "shard-burst" and index % 4 == 0:
            fill = palette.panel_secondary

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
                outline_width=random_source.uniform(0.0038, 0.0065),
                shadow_opacity=int(random_source.integers(20, 54)),
            )
        )

    return tuple(shapes)


def build_inner_shapes(
    random_source: Generator,
    *,
    palette: PaletteDefinition,
    center_radius: float,
) -> tuple[ShapeSpec, ...]:
    """Build one of several balanced symbol arrangements inside the center disk."""

    layouts: tuple[tuple[tuple[float, float, ShapeKind, float, float, float], ...], ...] = (
        (
            (-0.25, -0.20, "square", 0.18, 0.18, 0.0),
            (0.21, -0.10, "circle", 0.10, 0.10, 0.0),
            (0.15, 0.23, "square", 0.23, 0.23, 0.0),
            (-0.02, 0.12, "line", 0.28, 0.035, -42.0),
            (-0.13, 0.23, "square", 0.08, 0.08, 0.0),
        ),
        (
            (-0.20, 0.16, "square", 0.22, 0.22, 0.0),
            (0.03, 0.16, "square", 0.22, 0.22, 0.0),
            (0.25, 0.07, "circle", 0.12, 0.12, 0.0),
            (0.02, -0.19, "line", 0.30, 0.035, -44.0),
            (0.28, -0.18, "circle", 0.06, 0.06, 0.0),
        ),
        (
            (-0.22, -0.18, "circle", 0.20, 0.20, 0.0),
            (0.15, 0.05, "square", 0.25, 0.25, 0.0),
            (-0.05, 0.20, "square", 0.08, 0.08, 0.0),
            (0.24, -0.20, "circle", 0.07, 0.07, 0.0),
            (-0.02, 0.02, "line", 0.28, 0.035, -48.0),
        ),
        (
            (-0.26, -0.18, "square", 0.10, 0.10, 0.0),
            (0.22, -0.18, "circle", 0.08, 0.08, 0.0),
            (0.16, 0.18, "square", 0.20, 0.20, 0.0),
            (-0.08, 0.22, "square", 0.07, 0.07, 0.0),
            (0.00, 0.03, "line", 0.31, 0.035, -46.0),
        ),
    )
    layout = layouts[int(random_source.integers(0, len(layouts)))]

    shapes: list[ShapeSpec] = []
    for ox, oy, kind, width_scale, height_scale, rotation in layout:
        shapes.append(
            ShapeSpec(
                kind=kind,
                center_x=0.5 + ox * center_radius,
                center_y=0.5 + oy * center_radius,
                width=max(width_scale * center_radius, 0.008),
                height=max(height_scale * center_radius, 0.008),
                rotation_degrees=rotation,
                fill=palette.inner_fill,
                outline=palette.inner_outline,
                outline_width=0.0035,
                shadow_opacity=34,
            )
        )

    return tuple(shapes)
