"""Harmonic Waves: layered trigonometric fields with recipe-driven composition.

The recipe encodes rotation, warp strength, layer count, frequency set, and
blend mode. The renderer is a pure function of these values with no RNG calls.

Frequencies are selected from mathematically related sets to avoid chaotic
high-frequency interference while still producing varied compositions.
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

# Mathematically related frequency sets to prevent arbitrary high-freq interference.
_PHI = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio ≈ 1.618
_FREQUENCY_SETS: list[tuple[str, tuple[float, ...]]] = [
    ("harmonic",   (1.0, 2.0, 3.0, 4.0)),
    ("golden",     (1.0, _PHI, _PHI ** 2, _PHI ** 3)),
    ("sqrt-roots", (1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0)),
    ("pi-series",  (1.0, math.pi / 3.0, math.pi / 2.0, math.pi)),
]


class HarmonicWavesGenerator(SeededArrayGenerator):
    """Render flowing structures from interacting trigonometric fields."""

    @property
    def name(self) -> str:
        return "harmonic-waves"

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

        # --- Domain warp stages ---
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

        # --- Geometric parameters from geometry stream ---
        rotation = float(geom_rng.uniform(-math.pi, math.pi))
        warp_strength = float(geom_rng.uniform(0.10, 0.55))
        warp_freq_x = float(geom_rng.uniform(2.0, 6.0))
        warp_freq_y = float(geom_rng.uniform(2.0, 6.0))

        # Multiple phases (one per layer pair).
        phases = [float(geom_rng.uniform(0.0, math.tau)) for _ in range(layer_count + 2)]

        # Base frequencies scaled from the selected set.
        freq_scale = float(geom_rng.uniform(4.0, 8.0))

        # --- Palette ---
        palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = palette.name
        if request.options.palette_name:
            palette_name = request.options.palette_name

        palette_phase_shift = float(streams.lighting.uniform(0.0, 1.0))
        cosine_d = (
            palette.cosine_d[0] + palette_phase_shift,
            palette.cosine_d[1] + palette_phase_shift,
            palette.cosine_d[2] + palette_phase_shift,
        )

        # --- Rare events ---
        rare_events: list[str] = []
        if float(rarity_rng.random()) < 1 / 30:
            rare_events.append("luminous-halo")
        if float(rarity_rng.random()) < 1 / 80:
            rare_events.append("broken-symmetry")

        # --- Complexity / detail / modes ---
        complexity_level = (
            ComplexityLevel.COMPLEX if layer_count >= 5 else
            ComplexityLevel.MODERATE if layer_count >= 3 else
            ComplexityLevel.SIMPLE
        )

        params: dict[str, Any] = {
            "layer_count": layer_count,
            "warp_stages": warp_stages,
            "freq_set_name": freq_set_name,
            "rotation": rotation,
            "warp_strength": warp_strength,
            "warp_freq_x": warp_freq_x,
            "warp_freq_y": warp_freq_y,
            "freq_scale": freq_scale,
            "phases": phases,
            "palette_phase_d": list(cosine_d),
            "primary_frequency": freq_scale,  # used by compatibility rules
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
            detail_level=DetailLevel.MEDIUM,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.AMBIENT,
            accent_mode=(
                AccentMode.HIGHLIGHTS if "luminous-halo" in rare_events else AccentMode.NONE
            ),
            rare_events=tuple(rare_events),
            generator_params=params,
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render harmonic waves from a complete recipe. No RNG calls."""
        from pixel_forge.core.models.image_size import ImageSize

        size = ImageSize(width=recipe.width, height=recipe.height)
        field = build_coordinate_field(size)

        p = recipe.generator_params
        warp_stages = int(p["warp_stages"])
        freq_set_name = str(p["freq_set_name"])
        rotation = float(p["rotation"])
        warp_strength = float(p["warp_strength"])
        warp_freq_x = float(p["warp_freq_x"])
        warp_freq_y = float(p["warp_freq_y"])
        freq_scale = float(p["freq_scale"])
        phases: list[float] = list(p["phases"])
        palette_phase_d: list[float] = list(p["palette_phase_d"])
        layer_count = int(p["layer_count"])

        # Look up frequency set.
        freq_tuple: tuple[float, ...] = (1.0, _PHI, _PHI**2, 2.0)
        for name, fset in _FREQUENCY_SETS:
            if name == freq_set_name:
                freq_tuple = fset
                break

        # Rotate coordinate system.
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rx = field.x_centered * cos_r - field.y_centered * sin_r
        ry = field.x_centered * sin_r + field.y_centered * cos_r

        # Stage 1 domain warp.
        wx, wy = rx, ry
        if warp_stages >= 1:
            ph0 = phases[0] if len(phases) > 0 else 0.0
            ph1 = phases[1] if len(phases) > 1 else 0.0
            wx = rx + warp_strength * np.sin(ry * warp_freq_y + ph0)
            wy = ry + warp_strength * np.cos(rx * warp_freq_x + ph1)

        # Stage 2 domain warp.
        if warp_stages >= 2:
            ph2 = phases[2] if len(phases) > 2 else 0.0
            ph3 = phases[3] if len(phases) > 3 else 0.0
            s2 = warp_strength * 0.5
            wx = wx + s2 * np.cos(wy * warp_freq_y * 0.7 + ph2)
            wy = wy + s2 * np.sin(wx * warp_freq_x * 0.7 + ph3)

        r_warped = np.hypot(wx, wy)
        a_warped = np.arctan2(wy, wx)

        # Build combined field from layers.
        combined = np.zeros_like(wx)
        freqs = list(freq_tuple)
        layer_weights = [0.40, 0.30, 0.20, 0.10, 0.08, 0.06]

        for i in range(min(layer_count, len(freqs))):
            ph_i = phases[i % len(phases)] if phases else 0.0
            ph_j = phases[(i + 1) % len(phases)] if phases else 0.0
            f = freqs[i] * freq_scale
            wave = np.sin(
                wx * f + 1.4 * np.sin(wy * (f * 0.55) + ph_j) + ph_i
            )
            w = layer_weights[i] if i < len(layer_weights) else 0.05
            combined = combined + w * wave

        # Spiral component.
        ph_sp = phases[-2] if len(phases) >= 2 else 0.0
        spiral_freq = freq_scale * freqs[1] if len(freqs) > 1 else freq_scale
        spiral = np.sin(r_warped * spiral_freq + a_warped * 3.0 + ph_sp)
        combined = combined + 0.12 * spiral

        palette_position = 0.5 + 0.5 * np.sin(combined * 2.0 + r_warped * 0.5)
        brightness = 0.52 + 0.48 * (0.5 + 0.5 * np.cos(combined * math.pi - spiral * 0.6))

        # Look up palette for cosine coefficients.
        try:
            palette = _PALETTE_REGISTRY.get(recipe.palette_name)
            palette_c = palette.cosine_c
            phase_d = (
                float(palette_phase_d[0]) if len(palette_phase_d) > 0 else palette.cosine_d[0],
                float(palette_phase_d[1]) if len(palette_phase_d) > 1 else palette.cosine_d[1],
                float(palette_phase_d[2]) if len(palette_phase_d) > 2 else palette.cosine_d[2],
            )
        except Exception:
            palette_c = (1.0, 1.0, 1.0)
            phase_d = (float(palette_phase_d[0]) if palette_phase_d else 0.0,) * 3

        # Luminous halo accent.
        if "luminous-halo" in recipe.rare_events:
            halo_ring = np.exp(-((r_warped - 0.85) ** 2) / 0.005)
            brightness = np.clip(brightness + 0.5 * halo_ring, 0.0, 1.0)

        return cosine_palette_to_rgb_bytes(
            palette_position,
            phase=phase_d,
            frequency=palette_c,
            brightness=brightness,
        )
