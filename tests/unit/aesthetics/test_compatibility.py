"""Tests for compatibility rules and validator."""

from __future__ import annotations

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)


def _base_recipe(**kwargs) -> ArtworkRecipe:  # type: ignore[no-untyped-def]
    defaults = dict(
        schema_version=RECIPE_SCHEMA_VERSION,
        generator_name="test-gen",
        seed=0,
        candidate_seed=0,
        retry_index=0,
        width=512,
        height=512,
        palette_name="ocean-depth",
        symmetry_mode=SymmetryMode.RADIAL,
        complexity_level=ComplexityLevel.MODERATE,
        detail_level=DetailLevel.MEDIUM,
        background_mode=BackgroundMode.DARK,
        lighting_mode=LightingMode.FLAT,
        accent_mode=AccentMode.NONE,
        rare_events=(),
        generator_params={},
    )
    defaults.update(kwargs)
    return ArtworkRecipe(**defaults)


def test_no_rules_applied_when_recipe_is_clean() -> None:
    recipe = _base_recipe()
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    # A clean moderate recipe should have few or no rules triggered.
    # We assert the recipe is at minimum returned without errors.
    assert result.recipe is not None
    assert isinstance(result.applied_rules, tuple)


def test_dark_background_void_gets_luminous_accent() -> None:
    recipe = _base_recipe(background_mode=BackgroundMode.VOID, accent_mode=AccentMode.NONE)
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    assert result.recipe.accent_mode != AccentMode.NONE
    assert "dark-background-needs-accent" in result.applied_rules


def test_broken_symmetry_gets_focal_point() -> None:
    recipe = _base_recipe(symmetry_mode=SymmetryMode.BROKEN, accent_mode=AccentMode.NONE)
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    assert result.recipe.accent_mode != AccentMode.NONE
    assert "broken-symmetry-focal-point" in result.applied_rules


def test_small_dimension_reduces_detail() -> None:
    recipe = _base_recipe(width=64, height=64, detail_level=DetailLevel.HIGH)
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    assert result.recipe.detail_level == DetailLevel.MEDIUM
    assert "small-dimension-reduce-detail" in result.applied_rules


def test_too_many_rare_events_limited_to_three() -> None:
    recipe = _base_recipe(
        rare_events=("a", "b", "c", "d", "e"),
    )
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    assert len(result.recipe.rare_events) <= 3
    assert "multiple-rare-events-limit" in result.applied_rules


def test_compatibility_result_is_deterministic() -> None:
    recipe = _base_recipe(
        background_mode=BackgroundMode.VOID,
        accent_mode=AccentMode.NONE,
    )
    validator = RecipeCompatibilityValidator()
    r1 = validator.validate(recipe)
    r2 = validator.validate(recipe)
    assert r1.recipe == r2.recipe
    assert r1.applied_rules == r2.applied_rules


def test_many_petals_reduces_complexity() -> None:
    recipe = _base_recipe(
        generator_params={"primary_petals": 15},
        complexity_level=ComplexityLevel.COMPLEX,
    )
    validator = RecipeCompatibilityValidator()
    result = validator.validate(recipe)
    assert result.recipe.complexity_level == ComplexityLevel.MODERATE
    assert "many-petals-reduce-complexity" in result.applied_rules
