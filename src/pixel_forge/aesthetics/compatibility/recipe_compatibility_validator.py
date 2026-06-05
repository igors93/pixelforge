"""Compatibility validator that applies an ordered list of rules to a recipe.

Rules run in declaration order. Each rule receives the recipe returned by the
previous rule, so later rules see modifications made by earlier ones. No rule
introduces randomness; all decisions are deterministic given the recipe state.

Each rule's activation condition is designed to be reachable by the values
actually produced by the recipe builders. Thresholds are documented with the
range of values each generator can produce.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pixel_forge.aesthetics.compatibility.compatibility_rule import (
    CompatibilityResult,
    CompatibilityRule,
)
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)


class _HighFrequencySimplePaletteRule:
    """High geometric frequency forces a complexity downgrade.

    harmonic-waves produces primary_frequency in [4, 8] (freq_scale).
    Threshold at 6.5 is reachable by ~25% of harmonic-waves recipes.
    For INTRICATE recipes (layer_count ≥ 5), this reduces to COMPLEX.
    """

    @property
    def name(self) -> str:
        return "high-frequency-simple-palette"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        freq = float(recipe.generator_params.get("primary_frequency", 0.0))
        # Threshold 6.5 is within [4.0, 8.0] — reachable ~25% of the time.
        if freq > 6.5 and recipe.complexity_level == ComplexityLevel.INTRICATE:
            return _replace(recipe, complexity_level=ComplexityLevel.COMPLEX)
        return recipe


class _ManyPetalsReduceComplexityRule:
    """More than 12 radial petals reduces secondary-layer complexity.

    radial-bloom produces petals ∈ {13, 17} with combined 2% probability.
    Those recipes also set complexity_level=COMPLEX (petals > 9), so the
    condition fires for all 13- and 17-petal recipes.
    """

    @property
    def name(self) -> str:
        return "many-petals-reduce-complexity"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        petals = int(recipe.generator_params.get("primary_petals", 0))
        if petals > 12 and recipe.complexity_level in (
            ComplexityLevel.COMPLEX,
            ComplexityLevel.INTRICATE,
        ):
            return _replace(recipe, complexity_level=ComplexityLevel.MODERATE)
        return recipe


class _BrokenSymmetryFocalPointRule:
    """Broken symmetry keeps a stable central focal point (accent ≠ NONE)."""

    @property
    def name(self) -> str:
        return "broken-symmetry-focal-point"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if (
            recipe.symmetry_mode == SymmetryMode.BROKEN
            and recipe.accent_mode == AccentMode.NONE
        ):
            return _replace(recipe, accent_mode=AccentMode.HIGHLIGHTS)
        return recipe


class _DarkBackgroundNeedsAccentRule:
    """Very dark backgrounds require at least one visible luminous accent."""

    @property
    def name(self) -> str:
        return "dark-background-needs-accent"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if (
            recipe.background_mode == BackgroundMode.VOID
            and recipe.accent_mode == AccentMode.NONE
        ):
            return _replace(recipe, accent_mode=AccentMode.LUMINOUS)
        return recipe


class _DirectionalLightingDowngradeRule:
    """Directional lighting with a high layer count causes distracting shadows.

    Replaces the original _SaturatedPaletteReduceHighlightsRule which used
    palette_saturation_bias — a parameter no generator ever set.

    harmonic-waves with layer_count ≥ 5 and DIRECTIONAL lighting gets
    downgraded to AMBIENT so the many layers remain readable.
    """

    @property
    def name(self) -> str:
        return "directional-lighting-too-many-layers"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        layers = int(recipe.generator_params.get("layer_count", 0))
        if layers >= 5 and recipe.lighting_mode == LightingMode.DIRECTIONAL:
            return _replace(recipe, lighting_mode=LightingMode.AMBIENT)
        return recipe


class _SmallDimensionReduceDetailRule:
    """High detail is reduced for very small output dimensions."""

    @property
    def name(self) -> str:
        return "small-dimension-reduce-detail"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if (
            recipe.width < 128 or recipe.height < 128
        ) and recipe.detail_level == DetailLevel.HIGH:
            return _replace(recipe, detail_level=DetailLevel.MEDIUM)
        return recipe


class _DeepFractalZoomIterationsRule:
    """Deep fractal zoom requires more iterations for mandelbrot-dream.

    Zoom is in [zoom_min, zoom_max] per region — always reachable for high-zoom
    regions (double-spiral: 15–80, needle-region: 20–100, etc.).
    Fires when zoom > 5.0 and max_iterations < 256.
    """

    @property
    def name(self) -> str:
        return "deep-fractal-zoom-more-iterations"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if recipe.generator_name != "mandelbrot-dream":
            return recipe
        zoom = float(recipe.generator_params.get("zoom", 1.0))
        max_iter = int(recipe.generator_params.get("max_iterations", 100))
        if zoom > 5.0 and max_iter < 256:
            params = dict(recipe.generator_params)
            params["max_iterations"] = min(512, int(max_iter * zoom / 5.0))
            return _replace(recipe, generator_params=params)
        return recipe


class _StrongWarpReduceLayersRule:
    """Strong domain warping reduces the number of competing wave layers.

    harmonic-waves produces warp_strength in [0.10, 0.55].
    Threshold at 0.42 is reachable by ~20% of harmonic-waves recipes.
    Fires when warp > 0.42 and layer_count > 4.
    """

    @property
    def name(self) -> str:
        return "strong-warp-reduce-layers"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if recipe.generator_name != "harmonic-waves":
            return recipe
        warp = float(recipe.generator_params.get("warp_strength", 0.0))
        layers = int(recipe.generator_params.get("layer_count", 4))
        # Threshold 0.42 is within [0.10, 0.55] — reachable ~20% of the time.
        if warp > 0.42 and layers > 4:
            params = dict(recipe.generator_params)
            params["layer_count"] = 4
            return _replace(recipe, generator_params=params)
        return recipe


class _MultipleRareEventsLimitRule:
    """Multiple rare events must not create unreadable compositions."""

    @property
    def name(self) -> str:
        return "multiple-rare-events-limit"

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        if len(recipe.rare_events) > 3:
            return _replace(recipe, rare_events=recipe.rare_events[:3])
        return recipe


def _replace(recipe: ArtworkRecipe, **kwargs: Any) -> ArtworkRecipe:
    """Return a modified copy of a frozen recipe dataclass."""
    return replace(recipe, **kwargs)


_DEFAULT_RULES: tuple[CompatibilityRule, ...] = (
    _HighFrequencySimplePaletteRule(),
    _ManyPetalsReduceComplexityRule(),
    _BrokenSymmetryFocalPointRule(),
    _DarkBackgroundNeedsAccentRule(),
    _DirectionalLightingDowngradeRule(),
    _SmallDimensionReduceDetailRule(),
    _DeepFractalZoomIterationsRule(),
    _StrongWarpReduceLayersRule(),
    _MultipleRareEventsLimitRule(),
)


class RecipeCompatibilityValidator:
    """Apply all registered compatibility rules to a recipe in order."""

    def __init__(self, rules: tuple[CompatibilityRule, ...] = _DEFAULT_RULES) -> None:
        self._rules = rules

    def validate(self, recipe: ArtworkRecipe) -> CompatibilityResult:
        """Apply each rule and collect the names of rules that made changes."""
        current = recipe
        applied: list[str] = []

        for rule in self._rules:
            updated = rule.apply(current)
            if updated is not current:
                applied.append(rule.name)
                current = updated

        return CompatibilityResult(recipe=current, applied_rules=tuple(applied))
