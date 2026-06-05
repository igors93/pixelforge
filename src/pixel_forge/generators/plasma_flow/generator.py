"""Plasma Flow: coherent multi-scale field generator with recipe-driven pipeline.

Every recipe trait produces a measurable pixel difference:
  - palette_name      → full cosine_a/b/c/d coefficients
  - lighting_mode     → Gaussian / directional / ambient brightness modulation
  - background_mode   → edge vignette / light / void blend
  - accent_mode       → highlight / spark / luminous halo overlay
  - complexity_level  → warp stage cap
  - warp_stages       → domain warp depth
  - vortex_count      → curl distortions
  - flow_direction    → phase offset by direction
  - rare_events       → filament-surge and singularity visual transforms
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
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.plasma_flow.parameters import PlasmaFlowParams, VortexEntry
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()

# Complexity → max warp stages the renderer will apply.
_COMPLEXITY_WARP_CAP: dict[ComplexityLevel, int] = {
    ComplexityLevel.MINIMAL: 0,
    ComplexityLevel.SIMPLE: 1,
    ComplexityLevel.MODERATE: 1,
    ComplexityLevel.COMPLEX: 2,
    ComplexityLevel.INTRICATE: 2,
}

_RARE_EVENTS: dict[str, float] = {
    "filament-surge": 1 / 40,
    "singularity": 1 / 120,
}


class PlasmaFlowGenerator(SeededArrayGenerator):
    """Coherent plasma with domain warping and optional vortex distortion."""

    @property
    def name(self) -> str:
        return "plasma-flow"

    def render(self, request: GenerationRequest, random_source: np.random.Generator) -> UInt8Array:
        """Legacy render path."""
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
        geom_rng = streams.geometry
        palette_rng = streams.palette
        rarity_rng = streams.rarity

        trait_probs: list[TraitProbability] = []

        # --- Warp stages ---
        warp_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=0, weight=15.0),
            WeightedChoice(value=1, weight=55.0),
            WeightedChoice(value=2, weight=30.0),
        ]
        warp_result = sample_weighted(warp_choices, traits_rng)
        warp_stages = warp_result.value
        trait_probs.append(TraitProbability(
            trait_name="warp_stages",
            value=str(warp_stages),
            probability=warp_result.probability,
        ))

        # --- Vortex count ---
        vortex_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=0, weight=40.0),
            WeightedChoice(value=1, weight=35.0),
            WeightedChoice(value=2, weight=20.0),
            WeightedChoice(value=3, weight=5.0),
        ]
        vortex_result = sample_weighted(vortex_choices, traits_rng)
        vortex_count = vortex_result.value
        trait_probs.append(TraitProbability(
            trait_name="vortex_count",
            value=str(vortex_count),
            probability=vortex_result.probability,
        ))

        # --- Flow direction ---
        flow_choices: list[WeightedChoice[str]] = [
            WeightedChoice(value="radial", weight=35.0),
            WeightedChoice(value="diagonal", weight=30.0),
            WeightedChoice(value="horizontal", weight=20.0),
            WeightedChoice(value="turbulent", weight=15.0),
        ]
        flow_result = sample_weighted(flow_choices, traits_rng)
        flow_direction = flow_result.value
        trait_probs.append(TraitProbability(
            trait_name="flow_direction",
            value=flow_direction,
            probability=flow_result.probability,
        ))

        # --- Geometry ---
        phases = [float(geom_rng.uniform(0.0, math.tau)) for _ in range(6)]
        center_x = float(geom_rng.uniform(-0.40, 0.40))
        center_y = float(geom_rng.uniform(-0.40, 0.40))
        freq_low = float(geom_rng.uniform(7.0, 12.0))
        freq_high = float(geom_rng.uniform(14.0, 22.0))
        warp_strength = float(geom_rng.uniform(0.10, 0.45))
        turbulence = float(geom_rng.uniform(0.0, 0.30))

        vortex_entries: list[VortexEntry] = []
        for _ in range(vortex_count):
            vortex_entries.append(VortexEntry(
                x=float(geom_rng.uniform(-0.6, 0.6)),
                y=float(geom_rng.uniform(-0.6, 0.6)),
                strength=float(geom_rng.uniform(0.15, 0.45)),
                sign=1.0 if geom_rng.random() < 0.5 else -1.0,
            ))

        # --- Palette ---
        sampled_palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = request.options.palette_name or sampled_palette.name

        phase_shift = float(streams.lighting.uniform(0.0, 1.0))

        # --- Rare events ---
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

        # --- Derived recipe traits ---
        complexity_level = (
            ComplexityLevel.COMPLEX if warp_stages == 2 else
            ComplexityLevel.MODERATE if vortex_count > 0 else
            ComplexityLevel.SIMPLE
        )

        params = PlasmaFlowParams(
            warp_stages=warp_stages,
            vortex_count=vortex_count,
            flow_direction=flow_direction,
            phases=tuple(phases),
            center_x=center_x,
            center_y=center_y,
            freq_low=freq_low,
            freq_high=freq_high,
            warp_strength=warp_strength,
            turbulence=turbulence,
            vortex_data=tuple(vortex_entries),
            palette_phase_shift=phase_shift,
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
            symmetry_mode=SymmetryMode.NONE,
            complexity_level=complexity_level,
            detail_level=DetailLevel.MEDIUM,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.AMBIENT,
            accent_mode=AccentMode.NONE,
            rare_events=tuple(rare_events),
            generator_params=params.to_dict(),
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render plasma flow from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)
        p = PlasmaFlowParams.from_dict(recipe.generator_params)

        ph = list(p.phases) + [0.0] * 6  # pad for safe index access

        shifted_radius = np.hypot(
            field.x_centered - p.center_x,
            field.y_centered - p.center_y,
        )

        # Flow direction applies a phase bias to the base field.
        direction_bias = {
            "radial": 0.0,
            "diagonal": 0.5,
            "horizontal": 1.0,
            "turbulent": 1.5,
        }.get(p.flow_direction, 0.0)

        plasma = (
            np.sin(field.x_unit * p.freq_low * math.pi + ph[0] + direction_bias)
            + np.sin(field.y_unit * p.freq_low * math.pi + ph[1] + direction_bias * 0.7)
            + np.sin(
                (field.x_unit + field.y_unit) * p.freq_high * math.pi + ph[2] + direction_bias
            )
            + np.sin(shifted_radius * p.freq_high * 0.85 + ph[3])
        ) / 4.0

        # complexity_level caps the effective warp stages.
        warp_cap = _COMPLEXITY_WARP_CAP.get(recipe.complexity_level, 2)
        effective_warp = min(p.warp_stages, warp_cap)

        wx, wy = field.x_centered, field.y_centered
        if effective_warp >= 1:
            warp_y = p.warp_strength * np.sin(field.y_unit * p.freq_low * math.pi + ph[4])
            warp_x = p.warp_strength * np.cos(field.x_unit * p.freq_low * math.pi + ph[5])
            wx = field.x_centered + warp_y
            wy = field.y_centered + warp_x

        for vd in p.vortex_data:
            dx = wx - vd.x
            dy = wy - vd.y
            r2 = dx * dx + dy * dy + 0.01
            curl_x = -vd.sign * dy / r2 * vd.strength
            curl_y = vd.sign * dx / r2 * vd.strength
            vortex_r = np.hypot(
                wx + curl_x - (field.x_centered - p.center_x),
                wy + curl_y - (field.y_centered - p.center_y),
            )
            vortex_contribution = np.sin(vortex_r * p.freq_low * 2.0 + ph[0])
            plasma = plasma + 0.15 * vortex_contribution / max(len(p.vortex_data), 1)

        if effective_warp >= 2 and p.turbulence > 0:
            plasma = plasma + p.turbulence * np.sin(
                wx * p.freq_high * math.pi * 0.5 + wy * p.freq_high * math.pi * 0.5 + ph[2]
            )

        if "filament-surge" in recipe.rare_events:
            angle_field = np.arctan2(field.y_centered, field.x_centered)
            filament = np.sin(angle_field * 12.0 + shifted_radius * p.freq_high * 0.4) ** 8
            plasma = plasma + 0.20 * filament

        palette_position = np.clip(0.5 + 0.5 * plasma + 0.08 * shifted_radius, 0.0, 1.0)
        brightness = np.clip(
            0.46 + 0.54 * (0.5 + 0.5 * np.cos(plasma * math.pi + shifted_radius * 1.5)), 0.0, 1.0
        )

        if "singularity" in recipe.rare_events:
            singularity_mask = np.exp(-(shifted_radius**2) / 0.008)
            brightness = np.clip(brightness * (1.0 - 0.9 * singularity_mask), 0.0, 1.0)

        # Full cosine palette with palette phase shift baked into cosine_d.
        palette = _PALETTE_REGISTRY.get(recipe.palette_name)
        import math as _math

        import numpy as _np
        a = _np.asarray(palette.cosine_a, dtype=_np.float64)
        b = _np.asarray(palette.cosine_b, dtype=_np.float64)
        c = _np.asarray(palette.cosine_c, dtype=_np.float64)
        d = _np.asarray([
            palette.cosine_d[0] + p.palette_phase_shift,
            palette.cosine_d[1] + p.palette_phase_shift,
            palette.cosine_d[2] + p.palette_phase_shift,
        ], dtype=_np.float64)
        t = palette_position[..., _np.newaxis]
        rgb_float = a + b * _np.cos(_math.tau * (t * c + d))
        rgb_float = rgb_float * _np.clip(brightness, 0.0, 1.0)[..., _np.newaxis]

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=shifted_radius,
            x=field.x_centered,
            y=field.y_centered,
            lighting_mode=recipe.lighting_mode,
            background_mode=recipe.background_mode,
            accent_mode=recipe.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)

    def _get_recipe_params(self, recipe: ArtworkRecipe) -> dict[str, Any]:
        return dict(recipe.generator_params)
