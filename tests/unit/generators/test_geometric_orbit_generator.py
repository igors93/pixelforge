"""Regression and quality tests for the geometric-orbit generator."""

from pathlib import Path

import numpy as np

from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.generators.geometric_orbit import GeometricOrbitGenerator
from pixel_forge.generators.geometric_orbit.recipe_builder import build_recipe
from pixel_forge.generators.geometric_orbit.shape_grammar import (
    COMPOSITION_STYLES,
    SHAPE_GRAMMARS,
)


def _generate(seed: int, *, width: int = 128, height: int = 128) -> np.ndarray:
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

    assert len(unique_colors) > 120


def test_geometric_orbit_has_visible_contrast() -> None:
    image = _generate(9).astype(np.float64)
    luminance = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )

    contrast = float(
        np.percentile(luminance, 90) - np.percentile(luminance, 10)
    )
    assert contrast > 45.0


def test_geometric_orbit_supports_rectangular_output() -> None:
    image = _generate(11, width=160, height=96)

    assert image.shape == (96, 160, 3)


def test_all_shape_grammars_are_well_formed() -> None:
    assert set(COMPOSITION_STYLES) == set(SHAPE_GRAMMARS)

    for grammar in SHAPE_GRAMMARS.values():
        assert grammar.min_count > 0
        assert grammar.max_count >= grammar.min_count
        assert grammar.primary_shape != grammar.secondary_shape
        assert grammar.background_choices


def test_recipe_uses_only_its_shape_grammar() -> None:
    for seed in range(32):
        recipe = build_recipe(np.random.default_rng(seed))
        grammar = recipe.grammar
        allowed = {
            grammar.primary_shape,
            grammar.secondary_shape,
            grammar.tertiary_shape,
        }
        actual = {shape.kind for shape in recipe.outer_shapes}

        assert actual <= allowed


def test_seed_range_exposes_multiple_styles_and_palettes() -> None:
    recipes = [build_recipe(np.random.default_rng(seed)) for seed in range(48)]
    styles = {recipe.composition_style for recipe in recipes}
    palettes = {recipe.palette.name for recipe in recipes}

    assert len(styles) >= 6
    assert len(palettes) >= 8


def test_shapes_remain_inside_safe_canvas_region() -> None:
    for seed in range(24):
        recipe = build_recipe(np.random.default_rng(seed))
        for shape in recipe.outer_shapes:
            extent = max(shape.width, shape.height) * 0.72
            assert shape.center_x - extent >= 0.015
            assert shape.center_x + extent <= 0.985
            assert shape.center_y - extent >= 0.015
            assert shape.center_y + extent <= 0.985
