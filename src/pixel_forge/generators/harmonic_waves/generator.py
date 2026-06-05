"""Harmonic Waves: layered trigonometric fields with recipe-driven composition.

Every recipe trait produces a measurable pixel difference:
  - palette_name      → full cosine_a/b/c/d coefficients
  - symmetry_mode     → MIRROR_H/V folds coordinate axes; BROKEN shifts origin
  - complexity_level  → caps the active layer count (MINIMAL=2 … INTRICATE=6)
  - lighting_mode     → brightness modulation (radial / directional / ambient)
  - background_mode   → edge vignette / light / void / gradient blend
  - accent_mode       → highlight / spark / luminous overlay
  - layer_count       → number of frequency layers composited
  - warp_stages       → domain warp depth
  - freq_set_name     → which mathematical frequency set to use
  - rotation          → coordinate system rotation angle
  - warp_strength     → domain warp displacement magnitude
  - freq_scale        → base frequency multiplier
  - phases            → per-layer phase offsets
  - rare_events       → each produces a guaranteed visual transformation
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
from pixel_forge.generators.harmonic_waves.parameters import HarmonicWavesParams
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()

_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_FREQUENCY_SETS: list[tuple[str, tuple[float, ...]]] = [
    ("harmonic",   (1.0, 2.0, 3.0, 4.0)),
    ("golden",     (1.0, _PHI, _PHI ** 2, _PHI ** 3)),
    ("sqrt-roots", (1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0)),
    ("pi-series",  (1.0, math.pi / 3.0, math.pi / 2.0, math.pi)),
]

# Complexity level → maximum layer count.
_COMPLEXITY_LAYERS: dict[ComplexityLevel, int] = {
    ComplexityLevel.MINIMAL: 2,
    ComplexityLevel.SIMPLE: 3,
    ComplexityLevel.MODERATE: 4,
    ComplexityLevel.COMPLEX: 5,
    ComplexityLevel.INTRICATE: 6,
}

_RARE_EVENTS: dict[str, float] = {
    "luminous-halo": 1 / 30,
    "broken-symmetry": 1 / 80,
}


class HarmonicWavesGenerator(SeededArrayGenerator):
    """Render flowing structures from interacting trigonometric fields."""

    @property
    def name(self) -> str:
        return "harmonic-waves"

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
        comp_rng = streams.composition
        palette_rng = streams.palette
        rarity_rng = streams.rarity

        trait_probs: list[TraitProbability] = []

        # --- Layer count ---
        layer_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=2, weight=15.0),
            WeightedChoice(value=3, weight=30.0),
            WeightedChoice(value=4, weight=35.0),
            WeightedChoice(value=5, weight=15.0),
            WeightedChoice(value=6, weight=5.0),
        ]
        layer_result = sample_weighted(layer_choices, traits_rng)
        layer_count = layer_result.value
        trait_probs.append(TraitProbability(
            trait_name="layer_count",
            value=str(layer_count),
            probability=layer_result.probability,
        ))

        # --- Warp stages ---
        warp_choices: list[WeightedChoice[int]] = [
            WeightedChoice(value=0, weight=10.0),
            WeightedChoice(value=1, weight=55.0),
            WeightedChoice(value=2, weight=35.0),
        ]
        warp_result = sample_weighted(warp_choices, comp_rng)
        warp_stages = warp_result.value
        trait_probs.append(TraitProbability(
            trait_name="warp_stages",
            value=str(warp_stages),
            probability=warp_result.probability,
        ))

        # --- Frequency set ---
        freq_set_choices: list[WeightedChoice[str]] = [
            WeightedChoice(value="harmonic", weight=30.0),
            WeightedChoice(value="golden", weight=30.0),
            WeightedChoice(value="sqrt-roots", weight=25.0),
            WeightedChoice(value="pi-series", weight=15.0),
        ]
        freq_result = sample_weighted(freq_set_choices, geom_rng)
        freq_set_name = freq_result.value
        trait_probs.append(TraitProbability(
            trait_name="frequency_set",
            value=freq_set_name,
            probability=freq_result.probability,
        ))

        # --- Symmetry ---
        sym_choices: list[WeightedChoice[SymmetryMode]] = [
            WeightedChoice(value=SymmetryMode.NONE, weight=50.0),
            WeightedChoice(value=SymmetryMode.MIRROR_H, weight=25.0),
            WeightedChoice(value=SymmetryMode.BROKEN, weight=15.0),
            WeightedChoice(value=SymmetryMode.MIRROR_V, weight=10.0),
        ]
        sym_result = sample_weighted(sym_choices, traits_rng)
        symmetry_mode = sym_result.value
        trait_probs.append(TraitProbability(
            trait_name="symmetry_mode",
            value=symmetry_mode.value,
            probability=sym_result.probability,
        ))

        # --- Geometry parameters ---
        rotation = float(geom_rng.uniform(-math.pi, math.pi))
        # warp_strength in [0.10, 0.55]; compatibility rule fires at >0.40.
        warp_strength = float(geom_rng.uniform(0.10, 0.55))
        warp_freq_x = float(geom_rng.uniform(2.0, 6.0))
        warp_freq_y = float(geom_rng.uniform(2.0, 6.0))
        phases = [float(geom_rng.uniform(0.0, math.tau)) for _ in range(layer_count + 2)]
        freq_scale = float(geom_rng.uniform(4.0, 8.0))

        # --- Palette ---
        sampled_palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = request.options.palette_name or sampled_palette.name
        palette = _PALETTE_REGISTRY.get(palette_name)

        palette_phase_shift = float(streams.lighting.uniform(0.0, 1.0))
        cosine_d = (
            palette.cosine_d[0] + palette_phase_shift,
            palette.cosine_d[1] + palette_phase_shift,
            palette.cosine_d[2] + palette_phase_shift,
        )

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
            ComplexityLevel.COMPLEX if layer_count >= 5 else
            ComplexityLevel.MODERATE if layer_count >= 3 else
            ComplexityLevel.SIMPLE
        )

        params = HarmonicWavesParams(
            layer_count=layer_count,
            warp_stages=warp_stages,
            freq_set_name=freq_set_name,
            rotation=rotation,
            warp_strength=warp_strength,
            warp_freq_x=warp_freq_x,
            warp_freq_y=warp_freq_y,
            freq_scale=freq_scale,
            phases=tuple(phases),
            palette_phase_d=cosine_d,
            primary_frequency=freq_scale,
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
            detail_level=DetailLevel.MEDIUM,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.AMBIENT,
            accent_mode=(
                AccentMode.HIGHLIGHTS if "luminous-halo" in rare_events else AccentMode.NONE
            ),
            rare_events=tuple(rare_events),
            generator_params=params.to_dict(),
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render harmonic waves from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)
        p = HarmonicWavesParams.from_dict(recipe.generator_params)

        # Apply symmetry coordinate transform.
        xc, yc = apply_symmetry_to_coordinates(
            field.x_centered, field.y_centered, recipe.symmetry_mode
        )
        # BROKEN: slight origin shift.
        if recipe.symmetry_mode == SymmetryMode.BROKEN or "broken-symmetry" in recipe.rare_events:
            xc = field.x_centered + 0.10
            yc = field.y_centered - 0.08

        # Rotate coordinate system.
        cos_r = math.cos(p.rotation)
        sin_r = math.sin(p.rotation)
        rx = xc * cos_r - yc * sin_r
        ry = xc * sin_r + yc * cos_r

        # Domain warp stage 1.
        wx, wy = rx, ry
        phases = list(p.phases)
        ph = phases + [0.0] * 8  # safe index access
        if p.warp_stages >= 1:
            wx = rx + p.warp_strength * np.sin(ry * p.warp_freq_y + ph[0])
            wy = ry + p.warp_strength * np.cos(rx * p.warp_freq_x + ph[1])

        # Domain warp stage 2.
        if p.warp_stages >= 2:
            s2 = p.warp_strength * 0.5
            wx = wx + s2 * np.cos(wy * p.warp_freq_y * 0.7 + ph[2])
            wy = wy + s2 * np.sin(wx * p.warp_freq_x * 0.7 + ph[3])

        r_warped = np.hypot(wx, wy)
        a_warped = np.arctan2(wy, wx)

        # Resolve frequency set.
        freq_tuple: tuple[float, ...] = (1.0, _PHI, _PHI**2, 2.0)
        for name, fset in _FREQUENCY_SETS:
            if name == p.freq_set_name:
                freq_tuple = fset
                break

        # complexity_level caps the number of active layers.
        max_layers = _COMPLEXITY_LAYERS.get(recipe.complexity_level, 6)
        active_layers = min(p.layer_count, max_layers, len(freq_tuple))

        combined = np.zeros_like(wx)
        freqs = list(freq_tuple)
        layer_weights = [0.40, 0.30, 0.20, 0.10, 0.08, 0.06]

        for i in range(active_layers):
            ph_i = ph[i % len(ph)] if ph else 0.0
            ph_j = ph[(i + 1) % len(ph)] if ph else 0.0
            f = freqs[i] * p.freq_scale
            wave = np.sin(
                wx * f + 1.4 * np.sin(wy * (f * 0.55) + ph_j) + ph_i
            )
            w = layer_weights[i] if i < len(layer_weights) else 0.05
            combined = combined + w * wave

        # Spiral component.
        ph_sp = ph[-2] if len(ph) >= 2 else 0.0
        spiral_freq = p.freq_scale * freqs[1] if len(freqs) > 1 else p.freq_scale
        spiral = np.sin(r_warped * spiral_freq + a_warped * 3.0 + ph_sp)
        combined = combined + 0.12 * spiral

        palette_position = np.clip(0.5 + 0.5 * np.sin(combined * 2.0 + r_warped * 0.5), 0.0, 1.0)
        brightness = np.clip(
            0.52 + 0.48 * (0.5 + 0.5 * np.cos(combined * math.pi - spiral * 0.6)), 0.0, 1.0
        )

        # Luminous halo rare event.
        if "luminous-halo" in recipe.rare_events:
            halo_ring = np.exp(-((r_warped - 0.85) ** 2) / 0.005)
            brightness = np.clip(brightness + 0.5 * halo_ring, 0.0, 1.0)

        # Palette lookup and full cosine formula.
        palette = _PALETTE_REGISTRY.get(recipe.palette_name)
        # Blend the stored phase with the palette's own cosine_d.
        pd = p.palette_phase_d
        phase_d = (
            float(pd[0]) if pd else palette.cosine_d[0],
            float(pd[1]) if len(pd) > 1 else palette.cosine_d[1],
            float(pd[2]) if len(pd) > 2 else palette.cosine_d[2],
        )

        rgb_float = build_rgb_float_from_palette(palette_position, palette, brightness)
        # Override the phase with the stored shifted value so hue_shift is baked in.
        import math as _math

        import numpy as _np2
        a = _np2.asarray(palette.cosine_a, dtype=_np2.float64)
        b = _np2.asarray(palette.cosine_b, dtype=_np2.float64)
        c = _np2.asarray(palette.cosine_c, dtype=_np2.float64)
        d = _np2.asarray(phase_d, dtype=_np2.float64)
        t = palette_position[..., _np2.newaxis]
        rgb_float = a + b * _np2.cos(_math.tau * (t * c + d))
        if brightness is not None:
            rgb_float = rgb_float * _np2.clip(brightness, 0.0, 1.0)[..., _np2.newaxis]

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=r_warped,
            x=xc,
            y=yc,
            lighting_mode=recipe.lighting_mode,
            background_mode=recipe.background_mode,
            accent_mode=recipe.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)

    def _get_recipe_params(self, recipe: ArtworkRecipe) -> dict[str, Any]:
        return dict(recipe.generator_params)
