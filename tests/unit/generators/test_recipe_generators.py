"""Tests for recipe-driven generator pipeline: build_recipe + render_recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.generators.common.recipe_generator import RecipeGenerator
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
from pixel_forge.generators.radial_bloom import RadialBloomGenerator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams


def _request(seed: int = 42, width: int = 32, height: int = 24) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name="test",
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )


def _streams_and_seed(seed: int, generator_name: str) -> tuple[RandomStreams, int]:
    candidate_seed = derive_candidate_seed(
        master_seed=seed,
        generator_name=generator_name,
        retry_index=0,
        schema_version="1.0",
    )
    return RandomStreams.from_seed(candidate_seed), candidate_seed


@pytest.mark.parametrize(
    "generator",
    [
        HarmonicWavesGenerator(),
        PlasmaFlowGenerator(),
        RadialBloomGenerator(),
        MandelbrotDreamGenerator(),
    ],
)
def test_generator_implements_recipe_protocol(generator: object) -> None:
    assert isinstance(generator, RecipeGenerator)


@pytest.mark.parametrize(
    "generator",
    [
        HarmonicWavesGenerator(),
        PlasmaFlowGenerator(),
        RadialBloomGenerator(),
        MandelbrotDreamGenerator(),
    ],
)
def test_build_recipe_returns_recipe_and_trait_probs(generator: RecipeGenerator) -> None:
    req = _request()
    streams, candidate_seed = _streams_and_seed(42, generator.name)
    recipe, trait_probs = generator.build_recipe(req, streams, candidate_seed, retry_index=0)

    assert recipe.generator_name == generator.name
    assert recipe.width == 32
    assert recipe.height == 24
    assert recipe.seed >= 0
    assert isinstance(trait_probs, list)


@pytest.mark.parametrize(
    "generator",
    [
        HarmonicWavesGenerator(),
        PlasmaFlowGenerator(),
        RadialBloomGenerator(),
        MandelbrotDreamGenerator(),
    ],
)
def test_render_recipe_produces_correct_array_shape(generator: RecipeGenerator) -> None:
    req = _request(width=32, height=24)
    streams, candidate_seed = _streams_and_seed(42, generator.name)
    recipe, _ = generator.build_recipe(req, streams, candidate_seed, retry_index=0)
    rgb = generator.render_recipe(recipe)

    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize(
    "generator",
    [
        HarmonicWavesGenerator(),
        PlasmaFlowGenerator(),
        RadialBloomGenerator(),
        MandelbrotDreamGenerator(),
    ],
)
def test_render_recipe_is_deterministic(generator: RecipeGenerator) -> None:
    """The same recipe must produce identical pixel bytes every time."""
    req = _request(seed=123)
    streams, candidate_seed = _streams_and_seed(123, generator.name)
    recipe, _ = generator.build_recipe(req, streams, candidate_seed, retry_index=0)

    rgb1 = generator.render_recipe(recipe)
    rgb2 = generator.render_recipe(recipe)

    assert np.array_equal(rgb1, rgb2)


@pytest.mark.parametrize(
    "generator",
    [
        HarmonicWavesGenerator(),
        PlasmaFlowGenerator(),
        RadialBloomGenerator(),
        MandelbrotDreamGenerator(),
    ],
)
def test_different_seeds_produce_different_recipes(generator: RecipeGenerator) -> None:
    req1 = _request(seed=1)
    req2 = _request(seed=2)

    streams1, cs1 = _streams_and_seed(1, generator.name)
    streams2, cs2 = _streams_and_seed(2, generator.name)

    recipe1, _ = generator.build_recipe(req1, streams1, cs1, retry_index=0)
    recipe2, _ = generator.build_recipe(req2, streams2, cs2, retry_index=0)

    rgb1 = generator.render_recipe(recipe1)
    rgb2 = generator.render_recipe(recipe2)

    # Different seeds should produce different images (with very high probability).
    assert not np.array_equal(rgb1, rgb2)


def test_radial_bloom_petal_count_is_in_expected_range() -> None:
    """Petal count must always be within the declared weighted distribution."""
    generator = RadialBloomGenerator()
    valid_petal_counts = {5, 6, 7, 8, 9, 10, 11, 12, 13, 17}

    for seed in range(20):
        req = _request(seed=seed)
        streams, cs = _streams_and_seed(seed, generator.name)
        recipe, _ = generator.build_recipe(req, streams, cs, retry_index=0)
        petals = int(recipe.generator_params["primary_petals"])
        assert petals in valid_petal_counts


def test_recipe_stores_correct_generator_name() -> None:
    generator = RadialBloomGenerator()
    req = _request(seed=5)
    streams, cs = _streams_and_seed(5, generator.name)
    recipe, _ = generator.build_recipe(req, streams, cs, retry_index=0)
    assert recipe.generator_name == "radial-bloom"


def test_recipe_trait_probabilities_are_in_unit_range() -> None:
    generator = HarmonicWavesGenerator()
    req = _request(seed=7)
    streams, cs = _streams_and_seed(7, generator.name)
    _, trait_probs = generator.build_recipe(req, streams, cs, retry_index=0)

    for tp in trait_probs:
        assert 0.0 < tp.probability <= 1.0
        assert tp.information_bits >= 0.0
