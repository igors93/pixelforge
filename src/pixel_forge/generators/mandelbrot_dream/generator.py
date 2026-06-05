"""Mandelbrot Dream: named region-of-interest fractal with recipe pipeline.

Every recipe trait produces a measurable pixel difference:
  - palette_name   → palette cosine coefficients applied to escape-time coloring
  - detail_level   → LOW=100, MEDIUM=256, HIGH=512 max iterations
  - lighting_mode  → brightness modulation on the escape field
  - background_mode→ edge vignette / void blend
  - accent_mode    → highlight overlay on bright escape bands
  - region         → named Mandelbrot viewport
  - zoom           → logarithmic zoom level
  - interior_mode  → color of the non-escaping set
  - color_cycle    → hue cycle offset
  - rare_events    → perfect-alignment and golden-orbit transforms
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
from pixel_forge.generators.common.rendering import (
    apply_recipe_post_processing,
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.mandelbrot_dream.parameters import MandelbrotDreamParams
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted
from pixel_forge.rarity.trait_probability import TraitProbability

_PALETTE_REGISTRY = build_default_palette_registry()

# detail_level → max iteration count (renderer will pick the minimum of this and
# the zoom-scaled value set in build_recipe).
_DETAIL_MAX_ITER: dict[DetailLevel, int] = {
    DetailLevel.LOW: 100,
    DetailLevel.MEDIUM: 256,
    DetailLevel.HIGH: 512,
}

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

_RARE_EVENTS: dict[str, float] = {
    "perfect-alignment": 1 / 200,
    "golden-orbit": 1 / 500,
}


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

        # --- Region ---
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

        # --- Zoom ---
        zoom_min = float(region["zoom_min"])
        zoom_max = float(region["zoom_max"])
        log_zoom = float(geom_rng.uniform(math.log(zoom_min), math.log(zoom_max)))
        zoom = math.exp(log_zoom)

        # --- Iterations (zoom-scaled; detail_level will cap this) ---
        base_iterations = 100
        zoom_scaled_iter = int(min(512, base_iterations + int(zoom * 3.5)))

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

        color_cycle = float(geom_rng.uniform(0.0, 0.25))

        # --- Palette ---
        sampled_palette = _PALETTE_REGISTRY.sample(palette_rng, compatible_generator=self.name)
        palette_name = request.options.palette_name or sampled_palette.name

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
            ComplexityLevel.INTRICATE if zoom > 40.0 else
            ComplexityLevel.COMPLEX if zoom > 10.0 else
            ComplexityLevel.MODERATE
        )
        detail_level = DetailLevel.HIGH if zoom_scaled_iter > 200 else DetailLevel.MEDIUM

        params = MandelbrotDreamParams(
            region=str(region["name"]),
            zoom=zoom,
            center_real=center_real,
            center_imag=center_imag,
            max_iterations=zoom_scaled_iter,  # renderer caps via detail_level
            interior_mode=interior_mode,
            color_cycle=color_cycle,
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
            detail_level=detail_level,
            background_mode=BackgroundMode.DARK,
            lighting_mode=LightingMode.FLAT,
            accent_mode=AccentMode.NONE,
            rare_events=tuple(rare_events),
            generator_params=params.to_dict(),
        )
        return recipe, trait_probs

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render Mandelbrot fractal from a complete recipe. No RNG calls."""
        width = recipe.width
        height = recipe.height
        p = MandelbrotDreamParams.from_dict(recipe.generator_params)

        # detail_level caps the iteration count, ensuring visual difference.
        detail_cap = _DETAIL_MAX_ITER.get(recipe.detail_level, 256)
        max_iterations = min(p.max_iterations, detail_cap)

        aspect_ratio = width / height
        real_half = 1.6 / p.zoom * aspect_ratio
        imag_half = 1.6 / p.zoom

        real_axis = np.linspace(p.center_real - real_half, p.center_real + real_half, width)
        imag_axis = np.linspace(p.center_imag - imag_half, p.center_imag + imag_half, height)
        complex_plane = real_axis[None, :] + 1j * imag_axis[:, None]

        state = np.zeros_like(complex_plane)
        active = np.ones(complex_plane.shape, dtype=np.bool_)
        smooth_escape = np.zeros(complex_plane.shape, dtype=np.float64)

        for iteration in range(max_iterations):
            state[active] = state[active] * state[active] + complex_plane[active]
            escaped = (np.abs(state) > 2.0) & active
            if np.any(escaped):
                abs_values = np.abs(state[escaped])
                log2_abs = np.log2(np.maximum(abs_values, 2.0 + 1e-10))
                log2_log2_abs = np.log2(np.maximum(log2_abs, 1e-10))
                smooth_escape[escaped] = iteration + 1 - log2_log2_abs
                active[escaped] = False
            if not np.any(active):
                break

        escaped_mask = ~active
        normalized = np.zeros_like(smooth_escape)
        if np.any(escaped_mask):
            max_val = smooth_escape[escaped_mask].max()
            if max_val > 0:
                normalized[escaped_mask] = smooth_escape[escaped_mask] / max_val

        # Golden orbit rare event: highlight escape bands near 1/phi ≈ 0.618 in
        # normalized escape time. Uses normalized (not raw state magnitude) so the
        # condition is reachable for all fractal regions.
        if "golden-orbit" in recipe.rare_events:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            normalized_time = smooth_escape / max(max_iterations, 1)
            orbit_dist = np.abs(normalized_time - 1.0 / phi)
            orbit_ring = np.exp(-(orbit_dist**2) / 0.004)
            normalized = np.clip(normalized + 0.35 * orbit_ring * escaped_mask, 0.0, 1.0)

        # perfect-alignment: inject a symmetric ghosting pattern.
        if "perfect-alignment" in recipe.rare_events:
            sym_pattern = np.abs(np.sin(complex_plane.real * 8.0)) * 0.15
            normalized = np.clip(normalized + sym_pattern * escaped_mask, 0.0, 1.0)

        # Use the palette's cosine formula for escape-time coloring.
        palette = _PALETTE_REGISTRY.get(recipe.palette_name)
        palette_t = np.mod(p.color_cycle + 0.85 * normalized, 1.0)

        import math as _math

        import numpy as _np
        a = _np.asarray(palette.cosine_a, dtype=_np.float64)
        b = _np.asarray(palette.cosine_b, dtype=_np.float64)
        c = _np.asarray(palette.cosine_c, dtype=_np.float64)
        d = _np.asarray(palette.cosine_d, dtype=_np.float64)
        t = palette_t[..., _np.newaxis]
        rgb_float = a + b * _np.cos(_math.tau * (t * c + d))

        # Interior pixels use the palette's background color.
        if p.interior_mode == "black":
            interior_rgb = _np.array([0.04, 0.04, 0.04], dtype=_np.float64)
        elif p.interior_mode == "white":
            interior_rgb = _np.array([0.95, 0.95, 0.95], dtype=_np.float64)
        elif p.interior_mode == "nebula":
            interior_rgb = _np.asarray(palette.shadow_color, dtype=_np.float64) * 1.2
        else:  # dark-star
            interior_rgb = _np.asarray(palette.background_color, dtype=_np.float64)

        interior_mask = active[..., _np.newaxis]
        rgb_float = _np.where(interior_mask, interior_rgb, rgb_float)

        # Brightness ramp on escape field.
        brightness = _np.where(active, 0.0, 0.18 + 0.82 * normalized**0.65)
        rgb_float = rgb_float * _np.clip(brightness, 0.1, 1.0)[..., _np.newaxis]

        # Radius for post-processing (distance from center of image).
        h, w = height, width
        yi = (_np.arange(h) - h / 2.0) / max(min(h, w) / 2.0, 0.5)
        xi = (_np.arange(w) - w / 2.0) / max(min(h, w) / 2.0, 0.5)
        xx, yy = _np.meshgrid(xi, yi)
        radius = _np.hypot(xx, yy)

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=radius,
            x=xx,
            y=yy,
            lighting_mode=recipe.lighting_mode,
            background_mode=recipe.background_mode,
            accent_mode=recipe.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)

    def _get_recipe_params(self, recipe: ArtworkRecipe) -> dict[str, Any]:
        return dict(recipe.generator_params)
