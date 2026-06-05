"""Plasma Flow: coherent multi-scale field generator with recipe-driven pipeline.

Coherent value noise is synthesized from layered sine fields at different scales
and orientations, avoiding independent per-pixel random noise. Two-stage domain
warping and optional vortex distortions create organic, flowing plasma patterns.
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
from pixel_forge.generators.common.color import cosine_palette_to_rgb_bytes
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()


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

        # --- Geometric parameters ---
        phases = [float(geom_rng.uniform(0.0, math.tau)) for _ in range(6)]
        center_x = float(geom_rng.uniform(-0.40, 0.40))
        center_y = float(geom_rng.uniform(-0.40, 0.40))
        freq_low = float(geom_rng.uniform(7.0, 12.0))
        freq_high = float(geom_rng.uniform(14.0, 22.0))
        warp_strength = float(geom_rng.uniform(0.10, 0.45))
        turbulence = float(geom_rng.uniform(0.0, 0.30))

        # Vortex positions and strengths.
        vortex_data: list[dict[str, float]] = []
        for _ in range(vortex_count):
            vortex_data.append({
                "x": float(geom_rng.uniform(-0.6, 0.6)),
                "y": float(geom_rng.uniform(-0.6, 0.6)),
                "strength": float(geom_rng.uniform(0.15, 0.45)),
                "sign": 1.0 if geom_rng.random() < 0.5 else -1.0,
            })

        # --- Palette ---
        palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = palette.name
        if request.options.palette_name:
            palette_name = request.options.palette_name

        phase_shift = float(streams.lighting.uniform(0.0, 1.0))

        # --- Rare events ---
        rare_events: list[str] = []
        if float(rarity_rng.random()) < 1 / 40:
            rare_events.append("filament-surge")
        if float(rarity_rng.random()) < 1 / 120:
            rare_events.append("singularity")

        complexity_level = (
            ComplexityLevel.COMPLEX if warp_stages == 2 else
            ComplexityLevel.MODERATE if vortex_count > 0 else
            ComplexityLevel.SIMPLE
        )

        params: dict[str, Any] = {
            "warp_stages": warp_stages,
            "vortex_count": vortex_count,
            "flow_direction": flow_direction,
            "phases": phases,
            "center_x": center_x,
            "center_y": center_y,
            "freq_low": freq_low,
            "freq_high": freq_high,
            "warp_strength": warp_strength,
            "turbulence": turbulence,
            "vortex_data": vortex_data,
            "palette_phase_shift": phase_shift,
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
            symmetry_mode=SymmetryMode.NONE,
            complexity_level=complexity_level,
            detail_level=DetailLevel.MEDIUM,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.AMBIENT,
            accent_mode=AccentMode.NONE,
            rare_events=tuple(rare_events),
            generator_params=params,
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render plasma flow from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)

        p = recipe.generator_params
        warp_stages = int(p["warp_stages"])
        phases: list[float] = list(p["phases"])
        center_x = float(p["center_x"])
        center_y = float(p["center_y"])
        freq_low = float(p["freq_low"])
        freq_high = float(p["freq_high"])
        warp_strength = float(p["warp_strength"])
        turbulence = float(p["turbulence"])
        vortex_data: list[dict[str, float]] = list(p.get("vortex_data", []))
        phase_shift = float(p["palette_phase_shift"])

        ph = phases + [0.0] * 6  # pad so index access is safe

        shifted_radius = np.hypot(
            field.x_centered - center_x,
            field.y_centered - center_y,
        )

        # Base multi-scale sine field (coherent, not per-pixel random).
        plasma = (
            np.sin(field.x_unit * freq_low * math.pi + ph[0])
            + np.sin(field.y_unit * freq_low * math.pi + ph[1])
            + np.sin((field.x_unit + field.y_unit) * freq_high * math.pi + ph[2])
            + np.sin(shifted_radius * freq_high * 0.85 + ph[3])
        ) / 4.0

        # Domain warp (stage 1).
        wx, wy = field.x_centered, field.y_centered
        if warp_stages >= 1:
            warp_y = warp_strength * np.sin(field.y_unit * freq_low * math.pi + ph[4])
            warp_x = warp_strength * np.cos(field.x_unit * freq_low * math.pi + ph[5])
            wx = field.x_centered + warp_y
            wy = field.y_centered + warp_x

        # Vortex curl-like displacement.
        for vd in vortex_data:
            vx = float(vd["x"])
            vy = float(vd["y"])
            strength = float(vd["strength"])
            sign = float(vd["sign"])
            dx = wx - vx
            dy = wy - vy
            r2 = dx * dx + dy * dy + 0.01
            curl_x = -sign * dy / r2 * strength
            curl_y = sign * dx / r2 * strength
            r2_warped = np.hypot(wx + curl_x - (field.x_centered - center_x),
                                 wy + curl_y - (field.y_centered - center_y))
            vortex_contribution = np.sin(r2_warped * freq_low * 2.0 + ph[0])
            plasma = plasma + 0.15 * vortex_contribution / max(len(vortex_data), 1)

        # Stage 2 warp adds turbulence.
        if warp_stages >= 2 and turbulence > 0:
            plasma = plasma + turbulence * np.sin(
                wx * freq_high * math.pi * 0.5 + wy * freq_high * math.pi * 0.5 + ph[2]
            )

        # Filament surge rare event: amplify thin bright streaks.
        if "filament-surge" in recipe.rare_events:
            angle_field = np.arctan2(field.y_centered, field.x_centered)
            filament = np.sin(angle_field * 12.0 + shifted_radius * freq_high * 0.4) ** 8
            plasma = plasma + 0.20 * filament

        palette_position = np.clip(0.5 + 0.5 * plasma + 0.08 * shifted_radius, 0.0, 1.0)
        brightness = 0.46 + 0.54 * (0.5 + 0.5 * np.cos(plasma * math.pi + shifted_radius * 1.5))

        # Singularity: punch a dark hole at the center.
        if "singularity" in recipe.rare_events:
            singularity_mask = np.exp(-(shifted_radius**2) / 0.008)
            brightness = np.clip(brightness * (1.0 - 0.9 * singularity_mask), 0.0, 1.0)

        try:
            palette = _PALETTE_REGISTRY.get(recipe.palette_name)
            phase_d = (
                palette.cosine_d[0] + phase_shift,
                palette.cosine_d[1] + phase_shift,
                palette.cosine_d[2] + phase_shift,
            )
            palette_c = palette.cosine_c
        except Exception:
            phase_d = (phase_shift, phase_shift + 0.18, phase_shift + 0.52)
            palette_c = (1.0, 1.1, 0.9)

        return cosine_palette_to_rgb_bytes(
            palette_position,
            phase=phase_d,
            frequency=palette_c,
            brightness=brightness,
        )
