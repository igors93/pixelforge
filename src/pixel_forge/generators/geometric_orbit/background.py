"""Background panel generation for geometric-orbit."""

from __future__ import annotations

from numpy.random import Generator

from pixel_forge.generators.geometric_orbit.parameters import (
    BackgroundStyle,
    PaletteDefinition,
    PanelSpec,
)


def build_panels(
    random_source: Generator,
    *,
    style: BackgroundStyle,
    palette: PaletteDefinition,
) -> tuple[PanelSpec, ...]:
    """Create balanced large panels that stay behind the central composition."""

    configurations: dict[BackgroundStyle, tuple[int, tuple[float, float], tuple[float, float]]] = {
        "diagonal-panels": (5, (0.65, 1.25), (0.16, 0.30)),
        "angular-folds": (6, (0.48, 0.95), (0.18, 0.36)),
        "offset-ribbons": (4, (0.85, 1.35), (0.10, 0.20)),
        "split-field": (3, (0.90, 1.45), (0.30, 0.52)),
    }
    count, width_range, height_range = configurations[style]
    colors = (palette.panel_primary, palette.panel_secondary, palette.panel_accent)

    panels: list[PanelSpec] = []
    for index in range(count):
        if style == "split-field":
            rotation = (-38.0, 42.0, -12.0)[index % 3]
            center_x = (0.20, 0.78, 0.52)[index % 3]
            center_y = (0.26, 0.72, 0.50)[index % 3]
        else:
            rotation = random_source.uniform(-58.0, -24.0)
            if index % 2:
                rotation = random_source.uniform(24.0, 58.0)
            center_x = random_source.uniform(0.16, 0.84)
            center_y = random_source.uniform(0.12, 0.88)

        panels.append(
            PanelSpec(
                center_x=center_x,
                center_y=center_y,
                width=random_source.uniform(*width_range),
                height=random_source.uniform(*height_range),
                rotation_degrees=rotation,
                color=colors[index % len(colors)],
                corner_radius=random_source.uniform(0.01, 0.035),
                shadow_opacity=int(random_source.integers(22, 56)),
            )
        )

    return tuple(panels)
