"""Background construction for geometric-orbit."""

from __future__ import annotations

from numpy.random import Generator

from pixel_forge.generators.geometric_orbit.parameters import (
    BackgroundStyle,
    PaletteDefinition,
    PanelSpec,
    ShapeGrammar,
)


def build_panels(
    random_source: Generator,
    *,
    style: BackgroundStyle,
    palette: PaletteDefinition,
    grammar: ShapeGrammar,
) -> tuple[PanelSpec, ...]:
    """Create a small number of large panels that structure the composition."""

    if style == "quiet-field":
        count = 2
        width_range = (0.85, 1.25)
        height_range = (0.16, 0.28)
        opacity_range = (80, 145)
    elif style == "broad-ribbons":
        count = 4
        width_range = (0.95, 1.45)
        height_range = (0.16, 0.28)
        opacity_range = (150, 225)
    elif style == "crossing-beams":
        count = 5
        width_range = (0.80, 1.35)
        height_range = (0.10, 0.20)
        opacity_range = (170, 235)
    elif style == "angular-wedges":
        count = 4
        width_range = (0.70, 1.20)
        height_range = (0.24, 0.42)
        opacity_range = (145, 215)
    else:
        count = 3
        width_range = (1.00, 1.50)
        height_range = (0.30, 0.52)
        opacity_range = (175, 235)

    if grammar.name == "symbolic-minimal":
        count = min(count, 3)
    elif grammar.name in {"triangle-explosion", "geometric-sun"}:
        count = max(count, 4)

    colors = (
        palette.panel_primary,
        palette.panel_secondary,
        palette.panel_accent,
    )
    panels: list[PanelSpec] = []

    for index in range(count):
        if style == "split-diagonal":
            rotations = (-36.0, 42.0, -10.0)
            positions = ((0.18, 0.22), (0.80, 0.72), (0.52, 0.48))
            rotation = rotations[index % len(rotations)]
            center_x, center_y = positions[index % len(positions)]
        elif style == "crossing-beams":
            rotation = random_source.choice((-58.0, -42.0, 34.0, 49.0))
            center_x = random_source.uniform(0.12, 0.88)
            center_y = random_source.uniform(0.12, 0.88)
        elif style == "angular-wedges":
            rotation = random_source.uniform(-72.0, 72.0)
            center_x = random_source.uniform(0.12, 0.88)
            center_y = random_source.uniform(0.10, 0.90)
        else:
            rotation = random_source.uniform(-55.0, -25.0)
            if index % 2:
                rotation = random_source.uniform(25.0, 55.0)
            center_x = random_source.uniform(0.14, 0.86)
            center_y = random_source.uniform(0.12, 0.88)

        panels.append(
            PanelSpec(
                center_x=center_x,
                center_y=center_y,
                width=random_source.uniform(*width_range),
                height=random_source.uniform(*height_range),
                rotation_degrees=float(rotation),
                color=colors[index % len(colors)],
                corner_radius=random_source.uniform(0.008, 0.030),
                shadow_opacity=int(random_source.integers(28, 72)),
                opacity=int(random_source.integers(*opacity_range)),
            )
        )

    return tuple(panels)
