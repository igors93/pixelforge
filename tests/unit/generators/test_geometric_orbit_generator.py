"""Regression and quality tests for the geometric-orbit generator."""

from pathlib import Path

import numpy as np

from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.generators.geometric_orbit import GeometricOrbitGenerator


def _generate(seed: int, *, width: int = 96, height: int = 96) -> np.ndarray:
    generator = GeometricOrbitGenerator()
    image = generator.generate(
        GenerationRequest(
            size=ImageSize(width=width, height=height),
            generator_name="geometric-orbit",
            output_path=Path("unused.png"),
            seed=seed,
        )
    )
    return np.frombuffer(image.pixels, dtype=np.uint8).reshape(height, width, 3)


def test_geometric_orbit_is_reproducible() -> None:
    first = _generate(123)
    second = _generate(123)

    assert np.array_equal(first, second)


def test_geometric_orbit_changes_with_seed() -> None:
    first = _generate(1)
    second = _generate(2)

    assert not np.array_equal(first, second)


def test_geometric_orbit_has_meaningful_color_diversity() -> None:
    image = _generate(7)
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)

    assert len(unique_colors) > 80


def test_geometric_orbit_has_visible_contrast() -> None:
    image = _generate(9).astype(np.float64)
    luminance = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )

    assert float(np.percentile(luminance, 90) - np.percentile(luminance, 10)) > 35.0


def test_geometric_orbit_supports_rectangular_output() -> None:
    image = _generate(11, width=128, height=80)

    assert image.shape == (80, 128, 3)
