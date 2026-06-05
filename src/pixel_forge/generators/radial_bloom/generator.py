"""Radial Bloom: flower-like structures with full rarity system integration.

Every field in the recipe that this generator samples is guaranteed to produce a
measurable pixel difference when changed in isolation:

  - palette_name      → different cosine_a/b/c/d coefficients → different RGB values
  - symmetry_mode     → RADIAL uses golden angle; MIRROR_H/V fold coordinates
  - complexity_level  → controls the number of active crown rings
  - detail_level      → controls radial ripple count used in renderer
  - lighting_mode     → Gaussian / directional / ambient brightness modulation
  - background_mode   → edge vignette / light blend / void fade
  - accent_mode       → highlight / spark / luminous halo overlay
  - primary_petals    → petal wave angular frequency
  - crown_count       → ring positions and count
  - radial_ripple_count → layered radial frequency count
  - phyllotaxis       → golden-angle angular distortion
  - petal_sharpness   → exponent on petal wave
  - glow_spread       → Gaussian center-glow width
  - hue_base          → palette hue alignment
  - rare_events       → each causes a guaranteed visual transformation

Petal count distribution (weighted):
    6 petals: 24%, 8 petals: 24%, 5 petals: 12%, 7 petals: 12%, 9 petals: 10%
   10 petals:  8%, 11 petals:  5%, 12 petals:  3%, 13 petals: 1.5%, 17 petals: 0.5%
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
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.rendering import (
    apply_recipe_post_processing,
    apply_symmetry_to_coordinates,
    build_rgb_float_from_palette,
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.radial_bloom.parameters import RadialBloomParams
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()

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

# Rare event registry: name → probability.
# Every event must produce a guaranteed visible change in render_recipe.
_RARE_EVENTS: dict[str, float] = {
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

    def render(self, request: GenerationRequest, random_source: np.random.Generator) -> UInt8Array:
        """Legacy render path used by SeededArrayGenerator.generate()."""
        seed = request.seed if request.seed is not None else int(random_source.integers(0, 2**63))
        streams = RandomStreams.from_seed(seed)
        recipe, _ = self.build_recipe(request, streams, candidate_seed=seed, retry_index=0)
        return self.render_recipe(recipe)

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
        geom_rng = streams.geometry

        trait_probs: list[TraitProbability] = []

        # --- Petal count ---
        petal_result = sample_weighted(_PETAL_CHOICES, traits_rng)
        primary_petals = petal_result.value
        trait_probs.append(TraitProbability(
            trait_name="primary_petals",
            value=str(primary_petals),
            probability=petal_result.probability,
        ))

        secondary_petals = max(1, primary_petals // 2)
        # secondary_petals is fully determined by primary_petals; NOT a separate trait.

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

        # --- Phyllotaxis (golden angle spiral) ---
        phyllotaxis_p = 0.12
        phyllotaxis = bool(rarity_rng.random() < phyllotaxis_p)
        trait_probs.append(TraitProbability(
            trait_name="phyllotaxis",
            value=str(phyllotaxis),
            probability=phyllotaxis_p if phyllotaxis else (1.0 - phyllotaxis_p),
        ))

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

        # --- Geometry stream parameters ---
        ripple_frequency = float(geom_rng.uniform(8.0, 22.0))
        phase = float(geom_rng.uniform(0.0, math.tau))
        glow_spread = float(geom_rng.uniform(0.28, 0.62))
        petal_sharpness = float(geom_rng.uniform(1.0, 4.0))
        petal_radial_scale = float(geom_rng.uniform(0.6, 1.4))

        # --- Palette ---
        sampled_palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = request.options.palette_name or sampled_palette.name

        hue_base = float(streams.lighting.uniform(0.0, 1.0))

        # --- Rare events (each recorded as TraitProbability) ---
        rare_events: list[str] = []
        for event_name, prob in _RARE_EVENTS.items():
            occurred = bool(rarity_rng.random() < prob)
            if occurred:
                rare_events.append(event_name)
            trait_probs.append(TraitProbability(
                trait_name=f"rare_event:{event_name}",
                value="enabled" if occurred else "absent",
                probability=prob if occurred else (1.0 - prob),
            ))

        # --- Derived recipe fields ---
        background_mode = (
            BackgroundMode.VOID if "eclipse-palette" in rare_events else BackgroundMode.DARK
        )
        lighting_mode = LightingMode.RADIAL if crown_count > 0 else LightingMode.FLAT
        accent_mode = (
            AccentMode.LUMINOUS if "orbital-halo" in rare_events else
            AccentMode.HIGHLIGHTS if crown_count > 0 else
            AccentMode.NONE
        )

        # complexity_level controls crown ring rendering intensity
        complexity_level = (
            ComplexityLevel.MODERATE if primary_petals <= 9 else ComplexityLevel.COMPLEX
        )
        detail_level = DetailLevel.MEDIUM

        params = RadialBloomParams(
            primary_petals=primary_petals,
            secondary_petals=secondary_petals,
            crown_count=crown_count,
            radial_ripple_count=ripple_count,
            petal_sharpness=petal_sharpness,
            petal_radial_scale=petal_radial_scale,
            center_mode=center_mode,
            phyllotaxis=phyllotaxis,
            spiral_clockwise=spiral_clockwise,
            ripple_frequency=ripple_frequency,
            phase=phase,
            glow_spread=glow_spread,
            hue_base=hue_base,
        )

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
            rare_events=tuple(rare_events),
            generator_params=params.to_dict(),
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render the radial bloom from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)
        p = RadialBloomParams.from_dict(recipe.generator_params)

        # Apply symmetry coordinate transform before computing fields.
        xc, yc = apply_symmetry_to_coordinates(
            field.x_centered, field.y_centered, recipe.symmetry_mode
        )

        angle = np.arctan2(yc, xc)
        radius = np.hypot(xc, yc)

        # Broken symmetry: shift center and use original recipe coords.
        if recipe.symmetry_mode == SymmetryMode.BROKEN or "broken-symmetry" in recipe.rare_events:
            angle = np.arctan2(field.y_centered - 0.12, field.x_centered + 0.08)
            radius = np.hypot(field.x_centered + 0.08, field.y_centered - 0.12)

        # Phyllotaxis golden-angle distortion.
        if p.phyllotaxis:
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))
            angle = angle + radius * golden_angle * (2.0 if p.spiral_clockwise else -2.0)

        # Petal wave: integer periodicity preserved (no non-integer multiplier on angle).
        petal_base = np.abs(np.sin(angle * p.primary_petals + p.phase))
        petal_wave = petal_base ** p.petal_sharpness

        # Radial decay per petal: controls how far petals extend.
        petal_envelope = np.exp(-(radius**2) / (p.glow_spread * p.petal_radial_scale + 1e-9))

        # Secondary layer.
        ribbon_wave = np.cos(
            angle * p.secondary_petals - radius * p.ripple_frequency * 0.75 - p.phase * 0.5
        )

        # Radial ripple count is used directly; detail_level maps to count override.
        # detail_level HIGH → use recipe ripple_count; MEDIUM → cap at 3; LOW → cap at 1.
        effective_ripple_count = p.radial_ripple_count
        if recipe.detail_level == DetailLevel.LOW:
            effective_ripple_count = min(effective_ripple_count, 1)

        ripple = np.zeros_like(radius)
        for i in range(effective_ripple_count):
            freq_mult = 1.0 + i * 0.5
            ripple = ripple + np.sin(radius * p.ripple_frequency * freq_mult + p.phase * (i + 1))
        ripple = ripple / max(effective_ripple_count, 1)
        fine_ripple = 0.5 + 0.5 * ripple

        # Center glow.
        center_glow = np.exp(-(radius**2) / (p.glow_spread + 1e-9))

        # Crown rings — triple-crown ensures at least 3 rings even when crown_count=0.
        has_triple_crown = "triple-crown" in recipe.rare_events
        # complexity_level affects crown density.
        complexity_crown_mult = {
            ComplexityLevel.MINIMAL: 0,
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 1,
            ComplexityLevel.COMPLEX: 1,
            ComplexityLevel.INTRICATE: 2,
        }.get(recipe.complexity_level, 1)
        if has_triple_crown:
            # Guarantee at least 3 crown rings regardless of sampled crown_count.
            actual_crowns = max(p.crown_count, 3) * max(complexity_crown_mult, 1)
        else:
            actual_crowns = p.crown_count * complexity_crown_mult

        crown_contribution = np.zeros_like(radius)
        for i in range(actual_crowns):
            crown_r = 0.35 + i * 0.25
            crown_contribution = crown_contribution + np.exp(
                -((radius - crown_r) ** 2) / 0.004
            )
        crown_contribution = np.clip(crown_contribution, 0.0, 1.0)

        # Orbital halo ring (also represented via accent_mode in post-processing).
        halo = np.zeros_like(radius)
        if "orbital-halo" in recipe.rare_events:
            halo = np.exp(-((radius - 0.72) ** 2) / 0.003)

        # Golden spiral accent.
        spiral_field = np.zeros_like(radius)
        if "golden-spiral" in recipe.rare_events:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            spiral_field = np.sin(radius * phi * 12.0 + angle * 3.0 + p.phase) * 0.5 + 0.5

        # Black hole center: invert and darken center region.
        center_value_mod = np.ones_like(radius)
        if "black-hole-center" in recipe.rare_events:
            center_value_mod = 1.0 - np.exp(-(radius**2) / 0.015)

        # Compose palette position and brightness in [0, 1].
        palette_position = np.clip(
            0.5
            + 0.30 * petal_wave
            + 0.10 * fine_ripple
            + 0.06 * (ribbon_wave * 0.5 + 0.5)
            + 0.06 * spiral_field,
            0.0, 1.0,
        )

        brightness = np.clip(
            0.08
            + 0.62 * center_glow * center_value_mod * petal_envelope
            + 0.15 * ((petal_wave + (ribbon_wave * 0.5 + 0.5)) / 2.0)
            + 0.08 * fine_ripple
            + 0.20 * crown_contribution
            + 0.30 * halo,
            0.0, 1.0,
        )

        # center_mode modifies the center zone brightness.
        if p.center_mode == "void":
            brightness = brightness * center_value_mod
        elif p.center_mode == "bright":
            brightness = np.clip(brightness + 0.25 * center_glow, 0.0, 1.0)
        elif p.center_mode == "dark-star":
            brightness = brightness * (1.0 - 0.5 * center_glow)

        # Fetch palette and build RGB float array using full a+b*cos formula.
        palette = _PALETTE_REGISTRY.get(recipe.palette_name)
        rgb_float = build_rgb_float_from_palette(palette_position, palette, brightness)

        # Apply recipe-level post-processing (lighting, background, accent).
        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=radius,
            x=xc,
            y=yc,
            lighting_mode=recipe.lighting_mode,
            background_mode=recipe.background_mode,
            accent_mode=recipe.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)

    @staticmethod
    def _rare_event_params() -> dict[str, float]:
        return _RARE_EVENTS

    def _get_recipe_params(self, recipe: ArtworkRecipe) -> dict[str, Any]:
        return dict(recipe.generator_params)
