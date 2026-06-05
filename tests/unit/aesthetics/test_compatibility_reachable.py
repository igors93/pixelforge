"""Tests proving each compatibility rule can fire, what it changes, and that
non-activating recipes are left untouched.

Rules fire only when their condition is met — these tests construct both the
activating and non-activating recipe variants explicitly.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams

_VALIDATOR = RecipeCompatibilityValidator()


def _base_recipe(**overrides: object) -> ArtworkRecipe:
    defaults: dict[str, object] = dict(
        schema_version=RECIPE_SCHEMA_VERSION,
        generator_name="harmonic-waves",
        seed=1,
        candidate_seed=1,
        retry_index=0,
        width=256,
        height=256,
        palette_name="solar-flare",
        symmetry_mode=SymmetryMode.NONE,
        complexity_level=ComplexityLevel.MODERATE,
        detail_level=DetailLevel.MEDIUM,
        background_mode=BackgroundMode.DARK,
        lighting_mode=LightingMode.AMBIENT,
        accent_mode=AccentMode.NONE,
        rare_events=(),
        generator_params={},
    )
    defaults.update(overrides)
    return ArtworkRecipe(**defaults)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# high-frequency-simple-palette
# ─────────────────────────────────────────────────────────────────────────────

def test_high_frequency_rule_fires_for_intricate_high_freq() -> None:
    recipe = _base_recipe(
        complexity_level=ComplexityLevel.INTRICATE,
        generator_params={"primary_frequency": 7.0},
    )
    result = _VALIDATOR.validate(recipe)
    assert "high-frequency-simple-palette" in result.applied_rules
    assert result.recipe.complexity_level == ComplexityLevel.COMPLEX


def test_high_frequency_rule_does_not_fire_below_threshold() -> None:
    recipe = _base_recipe(
        complexity_level=ComplexityLevel.INTRICATE,
        generator_params={"primary_frequency": 5.0},
    )
    result = _VALIDATOR.validate(recipe)
    assert "high-frequency-simple-palette" not in result.applied_rules
    assert result.recipe.complexity_level == ComplexityLevel.INTRICATE


def test_high_frequency_rule_does_not_fire_for_non_intricate() -> None:
    recipe = _base_recipe(
        complexity_level=ComplexityLevel.COMPLEX,
        generator_params={"primary_frequency": 7.5},
    )
    result = _VALIDATOR.validate(recipe)
    assert "high-frequency-simple-palette" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# many-petals-reduce-complexity
# ─────────────────────────────────────────────────────────────────────────────

def test_many_petals_rule_fires_for_13_petals_complex() -> None:
    recipe = _base_recipe(
        generator_name="radial-bloom",
        complexity_level=ComplexityLevel.COMPLEX,
        generator_params={"primary_petals": 13},
    )
    result = _VALIDATOR.validate(recipe)
    assert "many-petals-reduce-complexity" in result.applied_rules
    assert result.recipe.complexity_level == ComplexityLevel.MODERATE


def test_many_petals_rule_does_not_fire_for_12_petals() -> None:
    recipe = _base_recipe(
        generator_name="radial-bloom",
        complexity_level=ComplexityLevel.COMPLEX,
        generator_params={"primary_petals": 12},
    )
    result = _VALIDATOR.validate(recipe)
    assert "many-petals-reduce-complexity" not in result.applied_rules


def test_many_petals_rule_does_not_fire_for_simple_complexity() -> None:
    recipe = _base_recipe(
        generator_name="radial-bloom",
        complexity_level=ComplexityLevel.SIMPLE,
        generator_params={"primary_petals": 17},
    )
    result = _VALIDATOR.validate(recipe)
    assert "many-petals-reduce-complexity" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# broken-symmetry-focal-point
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_symmetry_adds_highlights_accent() -> None:
    recipe = _base_recipe(symmetry_mode=SymmetryMode.BROKEN, accent_mode=AccentMode.NONE)
    result = _VALIDATOR.validate(recipe)
    assert "broken-symmetry-focal-point" in result.applied_rules
    assert result.recipe.accent_mode == AccentMode.HIGHLIGHTS


def test_broken_symmetry_does_not_override_existing_accent() -> None:
    recipe = _base_recipe(symmetry_mode=SymmetryMode.BROKEN, accent_mode=AccentMode.SPARKS)
    result = _VALIDATOR.validate(recipe)
    assert "broken-symmetry-focal-point" not in result.applied_rules
    assert result.recipe.accent_mode == AccentMode.SPARKS


def test_non_broken_symmetry_rule_does_not_fire() -> None:
    recipe = _base_recipe(symmetry_mode=SymmetryMode.NONE, accent_mode=AccentMode.NONE)
    result = _VALIDATOR.validate(recipe)
    assert "broken-symmetry-focal-point" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# dark-background-needs-accent
# ─────────────────────────────────────────────────────────────────────────────

def test_void_background_adds_luminous_accent() -> None:
    recipe = _base_recipe(background_mode=BackgroundMode.VOID, accent_mode=AccentMode.NONE)
    result = _VALIDATOR.validate(recipe)
    assert "dark-background-needs-accent" in result.applied_rules
    assert result.recipe.accent_mode == AccentMode.LUMINOUS


def test_void_background_with_existing_accent_no_change() -> None:
    recipe = _base_recipe(background_mode=BackgroundMode.VOID, accent_mode=AccentMode.HIGHLIGHTS)
    result = _VALIDATOR.validate(recipe)
    assert "dark-background-needs-accent" not in result.applied_rules


def test_dark_background_mode_does_not_trigger_void_rule() -> None:
    recipe = _base_recipe(background_mode=BackgroundMode.DARK, accent_mode=AccentMode.NONE)
    result = _VALIDATOR.validate(recipe)
    assert "dark-background-needs-accent" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# directional-lighting-too-many-layers
# ─────────────────────────────────────────────────────────────────────────────

def test_directional_lighting_downgraded_for_5_layers() -> None:
    recipe = _base_recipe(
        lighting_mode=LightingMode.DIRECTIONAL,
        generator_params={"layer_count": 5},
    )
    result = _VALIDATOR.validate(recipe)
    assert "directional-lighting-too-many-layers" in result.applied_rules
    assert result.recipe.lighting_mode == LightingMode.AMBIENT


def test_directional_lighting_kept_for_4_layers() -> None:
    recipe = _base_recipe(
        lighting_mode=LightingMode.DIRECTIONAL,
        generator_params={"layer_count": 4},
    )
    result = _VALIDATOR.validate(recipe)
    assert "directional-lighting-too-many-layers" not in result.applied_rules
    assert result.recipe.lighting_mode == LightingMode.DIRECTIONAL


def test_ambient_lighting_not_affected_by_layer_count() -> None:
    recipe = _base_recipe(
        lighting_mode=LightingMode.AMBIENT,
        generator_params={"layer_count": 6},
    )
    result = _VALIDATOR.validate(recipe)
    assert "directional-lighting-too-many-layers" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# small-dimension-reduce-detail
# ─────────────────────────────────────────────────────────────────────────────

def test_small_dimension_reduces_detail() -> None:
    recipe = _base_recipe(width=64, height=64, detail_level=DetailLevel.HIGH)
    result = _VALIDATOR.validate(recipe)
    assert "small-dimension-reduce-detail" in result.applied_rules
    assert result.recipe.detail_level == DetailLevel.MEDIUM


def test_large_dimension_keeps_high_detail() -> None:
    recipe = _base_recipe(width=512, height=512, detail_level=DetailLevel.HIGH)
    result = _VALIDATOR.validate(recipe)
    assert "small-dimension-reduce-detail" not in result.applied_rules


def test_medium_detail_not_affected_by_small_dimension() -> None:
    recipe = _base_recipe(width=64, height=64, detail_level=DetailLevel.MEDIUM)
    result = _VALIDATOR.validate(recipe)
    assert "small-dimension-reduce-detail" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# deep-fractal-zoom-more-iterations
# ─────────────────────────────────────────────────────────────────────────────

def test_deep_zoom_increases_iterations() -> None:
    recipe = _base_recipe(
        generator_name="mandelbrot-dream",
        generator_params={"zoom": 30.0, "max_iterations": 150},
    )
    result = _VALIDATOR.validate(recipe)
    assert "deep-fractal-zoom-more-iterations" in result.applied_rules
    assert int(result.recipe.generator_params["max_iterations"]) > 150


def test_deep_zoom_rule_skips_non_mandelbrot() -> None:
    recipe = _base_recipe(
        generator_name="harmonic-waves",
        generator_params={"zoom": 30.0, "max_iterations": 150},
    )
    result = _VALIDATOR.validate(recipe)
    assert "deep-fractal-zoom-more-iterations" not in result.applied_rules


def test_low_zoom_does_not_trigger_iteration_rule() -> None:
    recipe = _base_recipe(
        generator_name="mandelbrot-dream",
        generator_params={"zoom": 2.0, "max_iterations": 100},
    )
    result = _VALIDATOR.validate(recipe)
    assert "deep-fractal-zoom-more-iterations" not in result.applied_rules


def test_sufficient_iterations_not_boosted() -> None:
    recipe = _base_recipe(
        generator_name="mandelbrot-dream",
        generator_params={"zoom": 30.0, "max_iterations": 300},
    )
    result = _VALIDATOR.validate(recipe)
    assert "deep-fractal-zoom-more-iterations" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# strong-warp-reduce-layers
# ─────────────────────────────────────────────────────────────────────────────

def test_strong_warp_reduces_layer_count() -> None:
    recipe = _base_recipe(
        generator_name="harmonic-waves",
        generator_params={"warp_strength": 0.50, "layer_count": 6},
    )
    result = _VALIDATOR.validate(recipe)
    assert "strong-warp-reduce-layers" in result.applied_rules
    assert int(result.recipe.generator_params["layer_count"]) == 4


def test_moderate_warp_does_not_reduce_layers() -> None:
    recipe = _base_recipe(
        generator_name="harmonic-waves",
        generator_params={"warp_strength": 0.30, "layer_count": 6},
    )
    result = _VALIDATOR.validate(recipe)
    assert "strong-warp-reduce-layers" not in result.applied_rules


def test_strong_warp_rule_skips_non_harmonic() -> None:
    recipe = _base_recipe(
        generator_name="plasma-flow",
        generator_params={"warp_strength": 0.50, "layer_count": 6},
    )
    result = _VALIDATOR.validate(recipe)
    assert "strong-warp-reduce-layers" not in result.applied_rules


def test_strong_warp_4_layers_not_reduced() -> None:
    recipe = _base_recipe(
        generator_name="harmonic-waves",
        generator_params={"warp_strength": 0.50, "layer_count": 4},
    )
    result = _VALIDATOR.validate(recipe)
    assert "strong-warp-reduce-layers" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# multiple-rare-events-limit
# ─────────────────────────────────────────────────────────────────────────────

def test_more_than_3_rare_events_trimmed() -> None:
    recipe = _base_recipe(rare_events=("a", "b", "c", "d", "e"))
    result = _VALIDATOR.validate(recipe)
    assert "multiple-rare-events-limit" in result.applied_rules
    assert len(result.recipe.rare_events) == 3


def test_3_rare_events_not_trimmed() -> None:
    recipe = _base_recipe(rare_events=("a", "b", "c"))
    result = _VALIDATOR.validate(recipe)
    assert "multiple-rare-events-limit" not in result.applied_rules


def test_no_rare_events_not_affected() -> None:
    recipe = _base_recipe(rare_events=())
    result = _VALIDATOR.validate(recipe)
    assert "multiple-rare-events-limit" not in result.applied_rules


# ─────────────────────────────────────────────────────────────────────────────
# Rules affect rendered pixels (integration: rule fires → pixel difference)
# ─────────────────────────────────────────────────────────────────────────────

def _request(seed: int = 5, width: int = 32, height: int = 32) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name="harmonic-waves",
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )


def test_void_background_rule_changes_pixels() -> None:
    """dark-background-needs-accent: VOID+NONE → VOID+LUMINOUS adds a visible ring."""
    gen = HarmonicWavesGenerator()
    candidate_seed = derive_candidate_seed(
        master_seed=5, generator_name="harmonic-waves", retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    recipe, _ = gen.build_recipe(_request(), streams, candidate_seed=candidate_seed, retry_index=0)

    without_rule = replace(recipe, background_mode=BackgroundMode.VOID, accent_mode=AccentMode.NONE)
    result = _VALIDATOR.validate(without_rule)

    assert "dark-background-needs-accent" in result.applied_rules
    assert result.recipe.accent_mode == AccentMode.LUMINOUS

    pre_pixels = gen.render_recipe(without_rule)
    post_pixels = gen.render_recipe(result.recipe)
    assert not np.array_equal(pre_pixels, post_pixels)
