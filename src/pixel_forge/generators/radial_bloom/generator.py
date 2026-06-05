"""Radial Bloom: flower-like structures with full rarity system integration.

This generator is the primary showcase for the recipe-driven pipeline. Every
visual decision is made during build_recipe() and stored in the recipe; the
renderer is a pure function of the recipe with no random calls.

Petal count distribution (weighted):
    6 petals: 24%, 8 petals: 24%, 5 petals: 12%, 7 petals: 12%, 9 petals: 10%
   10 petals:  8%, 11 petals:  5%, 12 petals:  3%, 13 petals: 1.5%, 17 petals: 0.5%

Prime petal counts (5, 7, 11, 13, 17) influence rarity tier but do not override
the overall tier determined by the rarity evaluator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pixel_forge.aesthetics.palettes.palette_registry import build_default_palette_registry
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)
from pixel_forge.core.models.generation_request import GenerationRequest
from pixel_forge.generators.common.base import SeededArrayGenerator
from pixel_forge.generators.common.color import hsv_to_rgb_bytes
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

# Palette registry is module-level to avoid rebuilding it on every call.
_PALETTE_REGISTRY = build_default_palette_registry()

# Petal count weighted distribution (spec §13).
_PETAL_CHOICES: list[WeightedChoice[int]] = [
    WeightedChoice(value=6, weight=24.0),
    WeightedChoice(value=8, weight=24.0),
    WeightedChoice(value=5, weight=12.0),
    WeightedChoice(value=7, weight=12.0),
    WeightedChoice(value=9, weight=10.0),
    WeightedChoice(value=10, weight=8.0),
    WeightedChoice(value=11, weight=5.0),
    WeightedChoice(value=12, weight=3.0),
    WeightedChoice(value=13, weight=1.5),
    WeightedChoice(value=17, weight=0.5),
]
_PETAL_TOTAL = sum(c.weight for c in _PETAL_CHOICES)

# Rare event probabilities.
_RARE_EVENT_PROBS: dict[str, float] = {
    "triple-crown": 1 / 100,
    "black-hole-center": 1 / 150,
    "golden-spiral": 1 / 80,
    "orbital-halo": 1 / 50,
    "eclipse-palette": 1 / 250,
    "broken-symmetry": 1 / 60,
}


class RadialBloomGenerator(SeededArrayGenerator):
    """Petal-like structures with weighted trait sampling and rarity events."""

    @property
    def name(self) -> str:
        return "radial-bloom"

    # ------------------------------------------------------------------ #
    # Legacy path (backward compat): single-call generate → render.       #
    # The recipe pipeline is used by the updated GenerationService.       #
    # ------------------------------------------------------------------ #

    def render(self, request: GenerationRequest, random_source: np.random.Generator) -> UInt8Array:
        """Legacy render path used by SeededArrayGenerator.generate()."""
        from pixel_forge.randomness.random_streams import RandomStreams as RS

        seed = request.seed if request.seed is not None else int(random_source.integers(0, 2**63))
        streams = RS.from_seed(seed)
        recipe, _ = self.build_recipe(request, streams, candidate_seed=seed, retry_index=0)
        return self.render_recipe(recipe)

    # ------------------------------------------------------------------ #
    # Recipe-driven pipeline                                               #
    # ------------------------------------------------------------------ #

    def build_recipe(
        self,
        request: GenerationRequest,
        streams: RandomStreams,
        candidate_seed: int,
        retry_index: int,
    ) -> tuple[ArtworkRecipe, list[TraitProbability]]:
        traits_rng = streams.traits
        rarity_rng = streams.rarity
        palette_rng = streams.palette

        trait_probs: list[TraitProbability] = []

        # --- Petal count (weighted distribution) ---
        petal_result = sample_weighted(_PETAL_CHOICES, traits_rng)
        primary_petals = petal_result.value
        trait_probs.append(TraitProbability(
            trait_name="primary_petals",
            value=str(primary_petals),
            probability=petal_result.probability,
        ))

        secondary_petals = max(1, primary_petals // 2)

        # --- Crown count ---
        crown_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=0, weight=50.0),
            WeightedChoice(value=1, weight=30.0),
            WeightedChoice(value=2, weight=15.0),
            WeightedChoice(value=3, weight=5.0),
        ]
        crown_result = sample_weighted(crown_choices, traits_rng)
        crown_count = crown_result.value
        trait_probs.append(TraitProbability(
            trait_name="crown_count",
            value=str(crown_count),
            probability=crown_result.probability,
        ))

        # --- Radial ripple count ---
        ripple_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=1, weight=35.0),
            WeightedChoice(value=2, weight=40.0),
            WeightedChoice(value=3, weight=20.0),
            WeightedChoice(value=4, weight=5.0),
        ]
        ripple_result = sample_weighted(ripple_choices, traits_rng)
        ripple_count = ripple_result.value
        trait_probs.append(TraitProbability(
            trait_name="radial_ripple_count",
            value=str(ripple_count),
            probability=ripple_result.probability,
        ))

        # --- Symmetry ---
        symmetry_choices: list[WeightedChoice[SymmetryMode]] = [
            WeightedChoice(value=SymmetryMode.RADIAL, weight=60.0),
            WeightedChoice(value=SymmetryMode.MIRROR_H, weight=20.0),
            WeightedChoice(value=SymmetryMode.NONE, weight=12.0),
            WeightedChoice(value=SymmetryMode.BROKEN, weight=8.0),
        ]
        sym_result = sample_weighted(symmetry_choices, traits_rng)
        symmetry_mode = sym_result.value
        trait_probs.append(TraitProbability(
            trait_name="symmetry_mode",
            value=symmetry_mode.value,
            probability=sym_result.probability,
        ))

        # --- Phyllotaxis mode (golden angle spiral) ---
        phyllotaxis_p = 0.12
        phyllotaxis = bool(rarity_rng.random() < phyllotaxis_p)
        trait_probs.append(TraitProbability(
            trait_name="phyllotaxis",
            value=str(phyllotaxis),
            probability=phyllotaxis_p if phyllotaxis else (1.0 - phyllotaxis_p),
        ))

        # --- Spiral direction ---
        spiral_clockwise = bool(traits_rng.random() < 0.5)

        # --- Center mode ---
        center_choices: list[WeightedChoice[str]] = [
            WeightedChoice(value="glow", weight=50.0),
            WeightedChoice(value="void", weight=20.0),
            WeightedChoice(value="bright", weight=20.0),
            WeightedChoice(value="dark-star", weight=10.0),
        ]
        center_result = sample_weighted(center_choices, traits_rng)
        center_mode = center_result.value
        trait_probs.append(TraitProbability(
            trait_name="center_mode",
            value=center_mode,
            probability=center_result.probability,
        ))

        # --- Geometric parameters (from geometry stream) ---
        geom_rng = streams.geometry
        ripple_frequency = float(geom_rng.uniform(8.0, 22.0))
        phase = float(geom_rng.uniform(0.0, math.tau))
        glow_spread = float(geom_rng.uniform(0.28, 0.62))
        petal_width = float(geom_rng.uniform(0.6, 1.4))
        petal_curvature = float(geom_rng.uniform(0.4, 1.8))

        # --- Palette ---
        palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = palette.name
        # Override with user preference if provided.
        if request.options.palette_name:
            palette_name = request.options.palette_name

        # --- Hue base ---
        hue_base = float(streams.lighting.uniform(0.0, 1.0))

        # --- Rare events ---
        rare_events = self._sample_rare_events(rarity_rng)

        # --- Background mode (influenced by eclipse-palette rare event) ---
        background_mode = (
            BackgroundMode.VOID if "eclipse-palette" in rare_events else BackgroundMode.DARK
        )

        # --- Lighting mode ---
        lighting_mode = (
            LightingMode.RADIAL if crown_count > 0 else LightingMode.FLAT
        )

        # --- Accent mode ---
        accent_mode = (
            AccentMode.LUMINOUS if "orbital-halo" in rare_events else
            AccentMode.HIGHLIGHTS if crown_count > 0 else
            AccentMode.NONE
        )

        # --- Complexity and detail ---
        complexity_level = (
            ComplexityLevel.MODERATE if primary_petals <= 9 else ComplexityLevel.COMPLEX
        )
        detail_level = DetailLevel.MEDIUM

        params: dict[str, Any] = {
            "primary_petals": primary_petals,
            "secondary_petals": secondary_petals,
            "crown_count": crown_count,
            "radial_ripple_count": ripple_count,
            "petal_width": petal_width,
            "petal_curvature": petal_curvature,
            "center_mode": center_mode,
            "phyllotaxis": phyllotaxis,
            "spiral_clockwise": spiral_clockwise,
            "ripple_frequency": ripple_frequency,
            "phase": phase,
            "glow_spread": glow_spread,
            "hue_base": hue_base,
        }

        recipe = ArtworkRecipe(
            schema_version=RECIPE_SCHEMA_VERSION,
            generator_name=self.name,
            seed=request.seed if request.seed is not None else candidate_seed,
            candidate_seed=candidate_seed,
            retry_index=retry_index,
            width=request.size.width,
            height=request.size.height,
            palette_name=palette_name,
            symmetry_mode=symmetry_mode,
            complexity_level=complexity_level,
            detail_level=detail_level,
            background_mode=background_mode,
            lighting_mode=lighting_mode,
            accent_mode=accent_mode,
            rare_events=rare_events,
            generator_params=params,
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render the radial bloom from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)

        p = recipe.generator_params
        primary_petals = int(p["primary_petals"])
        secondary_petals = int(p["secondary_petals"])
        crown_count = int(p["crown_count"])
        ripple_count = int(p["radial_ripple_count"])
        petal_width = float(p["petal_width"])
        petal_curvature = float(p["petal_curvature"])
        phyllotaxis = bool(p["phyllotaxis"])
        spiral_clockwise = bool(p["spiral_clockwise"])
        ripple_freq = float(p["ripple_frequency"])
        phase = float(p["phase"])
        glow_spread = float(p["glow_spread"])
        hue_base = float(p["hue_base"])

        has_black_hole = "black-hole-center" in recipe.rare_events
        has_triple_crown = "triple-crown" in recipe.rare_events
        has_golden_spiral = "golden-spiral" in recipe.rare_events
        has_orbital_halo = "orbital-halo" in recipe.rare_events
        has_broken_sym = "broken-symmetry" in recipe.rare_events or (
            recipe.symmetry_mode == SymmetryMode.BROKEN
        )

        angle = field.angle
        radius = field.radius

        if has_broken_sym:
            # Shift the center slightly for broken symmetry.
            angle = np.arctan2(field.y_centered - 0.12, field.x_centered + 0.08)
            radius = np.hypot(field.x_centered + 0.08, field.y_centered - 0.12)

        if phyllotaxis:
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 137.5°
            angle = angle + radius * golden_angle * (2.0 if spiral_clockwise else -2.0)

        # Primary petal wave.
        petal_wave = np.sin(angle * primary_petals * petal_width + phase) ** 2
        petal_wave = petal_wave ** petal_curvature

        # Secondary layer.
        ribbon_wave = np.cos(
            angle * secondary_petals - radius * ripple_freq * 0.75 - phase * 0.5
        )

        # Radial ripples (multiple frequencies).
        ripple = np.zeros_like(radius)
        for i in range(ripple_count):
            freq_mult = 1.0 + i * 0.5
            ripple = ripple + np.sin(radius * ripple_freq * freq_mult + phase * (i + 1))
        ripple = ripple / max(ripple_count, 1)
        fine_ripple = 0.5 + 0.5 * ripple

        # Center glow.
        center_glow = np.exp(-(radius**2) / (glow_spread + 1e-9))

        # Crown rings (if any).
        crown_contribution = np.zeros_like(radius)
        actual_crowns = (crown_count * 2) if has_triple_crown else crown_count
        for i in range(actual_crowns):
            crown_r = 0.35 + i * 0.25
            crown_contribution = crown_contribution + np.exp(
                -((radius - crown_r) ** 2) / 0.004
            )
        crown_contribution = np.clip(crown_contribution, 0.0, 1.0)

        # Orbital halo ring.
        halo = np.zeros_like(radius)
        if has_orbital_halo:
            halo = np.exp(-((radius - 0.72) ** 2) / 0.003)

        # Golden spiral accent.
        spiral_field = np.zeros_like(radius)
        if has_golden_spiral:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            spiral_field = np.sin(radius * phi * 12.0 + angle * 3.0 + phase) * 0.5 + 0.5

        # Black hole center: invert and darken center region.
        center_value_mod = np.ones_like(radius)
        if has_black_hole:
            center_value_mod = 1.0 - np.exp(-(radius**2) / 0.015)

        # Compose hue, saturation, value.
        hue = (
            hue_base
            + 0.18 * petal_wave
            + 0.05 * fine_ripple
            + 0.04 * (ribbon_wave * 0.5 + 0.5)
        )

        saturation = np.clip(
            0.55 + 0.42 * center_glow + 0.08 * fine_ripple + 0.15 * crown_contribution,
            0.0, 1.0,
        )

        value = np.clip(
            0.08
            + 0.72 * center_glow * center_value_mod
            + 0.15 * ((petal_wave + (ribbon_wave * 0.5 + 0.5)) / 2.0)
            + 0.10 * fine_ripple
            + 0.20 * crown_contribution
            + 0.30 * halo
            + 0.08 * spiral_field,
            0.0, 1.0,
        )

        return hsv_to_rgb_bytes(hue, saturation, value)

    @staticmethod
    def _sample_rare_events(rng: np.random.Generator) -> tuple[str, ...]:
        """Sample rare visual events deterministically from the rarity stream."""
        events: list[str] = []
        for event_name, probability in _RARE_EVENT_PROBS.items():
            if rng.random() < probability:
                events.append(event_name)
        return tuple(events)
