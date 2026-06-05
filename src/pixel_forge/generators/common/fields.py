"""Coordinate-grid helpers used by mathematical image generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixel_forge.core.models import ImageSize
from pixel_forge.generators.common.types import FloatArray


@dataclass(frozen=True, slots=True)
class CoordinateField:
    """Precomputed Cartesian and polar coordinates for one image size."""

    x_unit: FloatArray
    y_unit: FloatArray
    x_centered: FloatArray
    y_centered: FloatArray
    radius: FloatArray
    angle: FloatArray


def build_coordinate_field(size: ImageSize) -> CoordinateField:
    """Create normalized coordinate arrays used by multiple generators.

    Unit coordinates cover the 0..1 range using pixel centers. Centered
    coordinates use the shortest image side as their scale, preserving circles
    and radial symmetry in both landscape and portrait images.
    """

    width = size.width
    height = size.height

    x_axis = (np.arange(width, dtype=np.float64) + 0.5) / width
    y_axis = (np.arange(height, dtype=np.float64) + 0.5) / height
    x_unit, y_unit = np.meshgrid(x_axis, y_axis)

    half_shortest_side = max(min(width, height) / 2.0, 0.5)
    x_centered = (
        np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
    ) / half_shortest_side
    y_centered = (
        np.arange(height, dtype=np.float64) + 0.5 - height / 2.0
    ) / half_shortest_side
    x_centered, y_centered = np.meshgrid(x_centered, y_centered)

    radius = np.hypot(x_centered, y_centered)
    angle = np.arctan2(y_centered, x_centered)

    return CoordinateField(
        x_unit=x_unit,
        y_unit=y_unit,
        x_centered=x_centered,
        y_centered=y_centered,
        radius=radius,
        angle=angle,
    )
