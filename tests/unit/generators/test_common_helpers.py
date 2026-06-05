import numpy as np
import pytest

from pixel_forge.core.models import ImageSize
from pixel_forge.generators.common import (
    build_coordinate_field,
    cosine_palette_to_rgb_bytes,
)


def test_coordinate_field_preserves_radial_scale_in_landscape_images() -> None:
    field = build_coordinate_field(ImageSize(width=200, height=100))

    horizontal_radius = field.radius[50, 149]
    vertical_radius = field.radius[99, 100]

    assert horizontal_radius == pytest.approx(vertical_radius, rel=0.03)


def test_cosine_palette_returns_rgb_bytes() -> None:
    values = np.linspace(0.0, 1.0, 12, dtype=np.float64).reshape(3, 4)

    rgb = cosine_palette_to_rgb_bytes(
        values,
        phase=(0.0, 0.2, 0.5),
    )

    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.uint8
