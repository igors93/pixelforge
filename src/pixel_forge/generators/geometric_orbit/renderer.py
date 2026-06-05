"""Pillow-based supersampled renderer for geometric-orbit."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixel_forge.core.models import ImageSize
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.geometric_orbit.parameters import GeometricOrbitParameters
from pixel_forge.generators.geometric_orbit.shapes import (
    draw_center_disk,
    draw_panel,
    draw_shape,
)

_SUPERSAMPLE = 2


class GeometricOrbitRenderer:
    """Render clean geometric artwork with antialiasing and safe composition."""

    def render(
        self,
        *,
        size: ImageSize,
        recipe: GeometricOrbitParameters,
    ) -> UInt8Array:
        scale = _SUPERSAMPLE
        width = size.width * scale
        height = size.height * scale
        shortest_side = min(width, height)

        canvas = Image.new("RGBA", (width, height), (*recipe.palette.canvas, 255))

        for panel in recipe.panels:
            draw_panel(
                canvas,
                center_x=round(panel.center_x * width),
                center_y=round(panel.center_y * height),
                width=max(8, round(panel.width * width)),
                height=max(8, round(panel.height * height)),
                rotation_degrees=panel.rotation_degrees,
                color=panel.color,
                corner_radius=max(2, round(panel.corner_radius * shortest_side)),
                shadow_color=recipe.palette.shadow,
                shadow_opacity=panel.shadow_opacity,
            )

        normalized_scale = shortest_side
        for shape in recipe.outer_shapes:
            draw_shape(
                canvas,
                shape=shape,
                scale=normalized_scale,
                shadow_color=recipe.palette.shadow,
            )

        center_x = width // 2
        center_y = height // 2
        center_radius = round(recipe.center_radius * shortest_side)
        draw_center_disk(
            canvas,
            center_x=center_x,
            center_y=center_y,
            radius=center_radius,
            fill=recipe.palette.center_fill,
            outline=recipe.palette.center_outline,
            highlight=recipe.palette.center_highlight,
            shadow=recipe.palette.shadow,
            outline_width=max(2, round(recipe.center_outline_width * shortest_side)),
            shadow_opacity=recipe.center_shadow_opacity,
            highlight_strength=recipe.center_highlight_strength,
        )

        for shape in recipe.inner_shapes:
            draw_shape(
                canvas,
                shape=shape,
                scale=normalized_scale,
                shadow_color=recipe.palette.shadow,
            )

        resized = canvas.convert("RGB").resize(
            (size.width, size.height),
            resample=Image.Resampling.LANCZOS,
        )
        array = np.asarray(resized, dtype=np.uint8)
        return np.ascontiguousarray(array)
