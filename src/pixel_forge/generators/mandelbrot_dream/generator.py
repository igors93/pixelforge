"""Mandelbrot Dream: named region-of-interest fractal with recipe pipeline.

Named regions focus the view on visually interesting areas of the Mandelbrot
set. Zoom is selected logarithmically and iteration limits scale with depth.
All parameters are stored in the recipe before rendering begins.
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
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()

# Named regions of interest with their visual centers and zoom ranges.
_REGIONS: list[dict[str, Any]] = [
    {
        "name": "main-cardioid",
        "center_real": -0.25, "center_imag": 0.0,
        "zoom_min": 0.8, "zoom_max": 1.5,
        "probability": 20.0,
    },
    {
        "name": "seahorse-valley",
        "center_real": -0.743, "center_imag": 0.127,
        "zoom_min": 6.0, "zoom_max": 30.0,
        "probability": 18.0,
    },
    {
        "name": "elephant-valley",
        "center_real": 0.300, "center_imag": 0.0,
        "zoom_min": 4.0, "zoom_max": 20.0,
        "probability": 15.0,
    },
    {
        "name": "double-spiral",
        "center_real": -0.748, "center_imag": 0.100,
        "zoom_min": 15.0, "zoom_max": 80.0,
        "probability": 12.0,
    },
    {
        "name": "mini-mandelbrot",
        "center_real": -1.750, "center_imag": 0.0,
        "zoom_min": 8.0, "zoom_max": 40.0,
        "probability": 12.0,
    },
    {
        "name": "needle-region",
        "center_real": -1.77, "center_imag": 0.0,
        "zoom_min": 20.0, "zoom_max": 100.0,
        "probability": 8.0,
    },
    {
        "name": "satellite-islands",
        "center_real": -0.156, "center_imag": 1.031,
        "zoom_min": 4.0, "zoom_max": 25.0,
        "probability": 8.0,
    },
    {
        "name": "spiral-junction",
        "center_real": -0.722, "center_imag": 0.246,
        "zoom_min": 8.0, "zoom_max": 50.0,
        "probability": 7.0,
    },
]


class MandelbrotDreamGenerator(SeededArrayGenerator):
    """Mandelbrot set with named regions, logarithmic zoom, and recipe pipeline."""

    @property
    def name(self) -> str:
        return "mandelbrot-dream"

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

        # --- Region of interest ---
        region_choices: list[WeightedChoice[dict[str, Any]]] = [
            WeightedChoice(value=r, weight=float(r["probability"])) for r in _REGIONS
        ]
        region_result = sample_weighted(region_choices, traits_rng)
        region: dict[str, Any] = region_result.value
        trait_probs.append(TraitProbability(
            trait_name="region",
            value=str(region["name"]),
            probability=region_result.probability,
        ))

        # --- Logarithmic zoom selection ---
        zoom_min = float(region["zoom_min"])
        zoom_max = float(region["zoom_max"])
        log_zoom = float(geom_rng.uniform(math.log(zoom_min), math.log(zoom_max)))
        zoom = math.exp(log_zoom)

        # Iteration count scales with zoom depth; deeper zoom needs more iterations.
        base_iterations = 100
        max_iterations = int(min(512, base_iterations + int(zoom * 3.5)))

        # Center offset jitter (small random displacement from the region center).
        center_jitter = 0.05 / max(zoom * 0.1, 1.0)
        jitter_r = float(geom_rng.uniform(-center_jitter, center_jitter))
        jitter_i = float(geom_rng.uniform(-center_jitter, center_jitter))
        center_real = float(region["center_real"]) + jitter_r
        center_imag = float(region["center_imag"]) + jitter_i

        # --- Interior mode ---
        interior_choices: list[WeightedChoice[str]] = [
            WeightedChoice(value="black", weight=40.0),
            WeightedChoice(value="nebula", weight=35.0),
            WeightedChoice(value="white", weight=15.0),
            WeightedChoice(value="dark-star", weight=10.0),
        ]
        interior_result = sample_weighted(interior_choices, traits_rng)
        interior_mode = interior_result.value
        trait_probs.append(TraitProbability(
            trait_name="interior_mode",
            value=interior_mode,
            probability=interior_result.probability,
        ))

        # --- Color cycle direction ---
        color_cycle = float(geom_rng.uniform(0.0, 0.25))

        # --- Palette ---
        palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = palette.name
        if request.options.palette_name:
            palette_name = request.options.palette_name

        # --- Rare events ---
        rare_events: list[str] = []
        if float(rarity_rng.random()) < 1 / 200:
            rare_events.append("perfect-alignment")
        if float(rarity_rng.random()) < 1 / 500:
            rare_events.append("golden-orbit")

        complexity_level = (
            ComplexityLevel.INTRICATE if zoom > 40.0 else
            ComplexityLevel.COMPLEX if zoom > 10.0 else
            ComplexityLevel.MODERATE
        )

        detail_level = DetailLevel.HIGH if max_iterations > 200 else DetailLevel.MEDIUM

        params: dict[str, Any] = {
            "region": str(region["name"]),
            "zoom": zoom,
            "center_real": center_real,
            "center_imag": center_imag,
            "max_iterations": max_iterations,
            "interior_mode": interior_mode,
            "color_cycle": color_cycle,
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
            detail_level=detail_level,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.FLAT,
            accent_mode=AccentMode.NONE,
            rare_events=tuple(rare_events),
            generator_params=params,
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render Mandelbrot fractal from a complete recipe. No RNG calls."""
        width = recipe.width
        height = recipe.height

        p = recipe.generator_params
        zoom = float(p["zoom"])
        center_real = float(p["center_real"])
        center_imag = float(p["center_imag"])
        max_iterations = int(p["max_iterations"])
        interior_mode = str(p["interior_mode"])
        color_cycle = float(p["color_cycle"])

        aspect_ratio = width / height
        real_half = 1.6 / zoom * aspect_ratio
        imag_half = 1.6 / zoom

        real_axis = np.linspace(center_real - real_half, center_real + real_half, width)
        imag_axis = np.linspace(center_imag - imag_half, center_imag + imag_half, height)
        complex_plane = real_axis[None, :] + 1j * imag_axis[:, None]

        state = np.zeros_like(complex_plane)
        active = np.ones(complex_plane.shape, dtype=np.bool_)
        smooth_escape = np.zeros(complex_plane.shape, dtype=np.float64)

        for iteration in range(max_iterations):
            state[active] = state[active] * state[active] + complex_plane[active]
            escaped = (np.abs(state) > 2.0) & active
            if np.any(escaped):
                # Smooth escape-time coloring using the bailout value.
                abs_values = np.abs(state[escaped])
                # Protect log against values exactly at 2.0.
                log2_abs = np.log2(np.maximum(abs_values, 2.0 + 1e-10))
                smooth_escape[escaped] = (
                    iteration + 1 - np.log2(np.maximum(log2_abs, 1e-10))
                )
                active[escaped] = False
            if not np.any(active):
                break

        escaped_mask = ~active
        normalized = np.zeros_like(smooth_escape)
        if np.any(escaped_mask):
            max_val = smooth_escape[escaped_mask].max()
            if max_val > 0:
                normalized[escaped_mask] = smooth_escape[escaped_mask] / max_val

        hue = color_cycle + 0.85 * normalized

        # Interior color depends on interior_mode.
        if interior_mode == "black":
            sat_interior, val_interior = 0.0, 0.04
        elif interior_mode == "white":
            sat_interior, val_interior = 0.0, 0.95
        elif interior_mode == "nebula":
            sat_interior, val_interior = 0.55, 0.20
        else:  # dark-star
            sat_interior, val_interior = 0.30, 0.08

        saturation = np.where(active, sat_interior, 0.90)
        value = np.where(active, val_interior, 0.18 + 0.82 * normalized**0.65)

        # Golden orbit rare event: show orbit-trap circles.
        if "golden-orbit" in recipe.rare_events:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            orbit_dist = np.abs(np.abs(state) - phi)
            orbit_ring = np.exp(-(orbit_dist**2) / 0.003)
            value = np.clip(value + 0.4 * orbit_ring * escaped_mask, 0.0, 1.0)
            hue = hue + 0.08 * orbit_ring * escaped_mask

        return hsv_to_rgb_bytes(hue, saturation, value)
