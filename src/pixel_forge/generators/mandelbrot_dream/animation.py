"""Animated renderer for the Mandelbrot Dream generator.

Conservative loop-safe animation only — no infinite zoom. All transformations
remain within the sampled region of interest and use periodic math exclusively.

Motion profiles:
  color-cycle    – color_cycle offset advances by integer cycles (default)
  micro-orbit    – camera centre performs a tiny circular orbit
  fractal-breath – zoom pulsates slightly around the recipe zoom value
  orbit-trap-cycle – orbit-trap phase shifts (when golden-orbit is active)

Rare animation events:
  golden-orbit-motion – the golden-orbit ring oscillates in normalized time
  perfect-loop        – perfect-alignment ghosting pattern rotates
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pixel_forge.animation.animation_randomness import AnimationStreams
from pixel_forge.animation.frame_timing import frame_duration_ms_from_fps
from pixel_forge.animation.loop_math import (
    circular_orbit,
    cyclic_sine,
    periodic_color_shift,
    smooth_periodic_envelope,
)
from pixel_forge.animation.motion_profiles import (
    MANDELBROT_DEFAULT_PROFILE,
    MANDELBROT_PROFILES,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION, AnimationRecipe
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.generators.common.rendering import apply_recipe_post_processing, rgb_float_to_bytes
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.mandelbrot_dream.generator import (
    _DETAIL_MAX_ITER,
    _PALETTE_REGISTRY,
)
from pixel_forge.generators.mandelbrot_dream.parameters import MandelbrotDreamParams
from pixel_forge.rarity.trait_probability import TraitProbability

_RARE_ANIMATION_EVENTS: dict[str, float] = {
    "golden-orbit-motion": 1 / 15,
    "perfect-loop": 1 / 10,
}


@dataclass(frozen=True, slots=True)
class MandelbrotAnimationParams:
    """Generator-specific animation parameters for mandelbrot-dream."""

    motion_profile: str
    color_cycles: int           # integer palette cycles per loop
    orbit_radius: float         # camera micro-orbit radius (fraction of viewport)
    zoom_pulse_amplitude: float # fractional zoom pulse amplitude (0 → no pulse)
    golden_orbit_shift: bool    # rare: oscillate golden-orbit normalized time

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_profile": self.motion_profile,
            "color_cycles": self.color_cycles,
            "orbit_radius": self.orbit_radius,
            "zoom_pulse_amplitude": self.zoom_pulse_amplitude,
            "golden_orbit_shift": self.golden_orbit_shift,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MandelbrotAnimationParams:
        return cls(
            motion_profile=str(d["motion_profile"]),
            color_cycles=int(d["color_cycles"]),
            orbit_radius=float(d["orbit_radius"]),
            zoom_pulse_amplitude=float(d["zoom_pulse_amplitude"]),
            golden_orbit_shift=bool(d["golden_orbit_shift"]),
        )


class MandelbrotDreamAnimator:
    """Conservative animated renderer for mandelbrot-dream recipes."""

    @property
    def name(self) -> str:
        return "mandelbrot-dream"

    def build_animation_recipe(
        self,
        artwork_recipe: ArtworkRecipe,
        options: AnimationOptions,
        streams: AnimationStreams,
        animation_seed: int,
    ) -> tuple[AnimationRecipe, list[TraitProbability]]:
        trait_probs: list[TraitProbability] = []

        profile = options.motion_profile
        if profile is None or profile not in MANDELBROT_PROFILES:
            profile = MANDELBROT_DEFAULT_PROFILE

        intensity = options.motion_intensity
        if intensity is None:
            intensity = float(streams.motion.uniform(0.5, 1.0))

        color_cycles = int(streams.color.integers(1, 4))
        # Micro-orbit radius as a tiny fraction of the viewport half-width.
        orbit_radius = float(streams.camera.uniform(0.002, 0.012))
        # Zoom pulse: ≤ 5% of recipe zoom to stay within the region.
        zoom_pulse_amplitude = float(streams.camera.uniform(0.01, 0.05)) * intensity

        rare_anim_events: list[str] = []
        golden_orbit_shift = False
        for event_name, prob in _RARE_ANIMATION_EVENTS.items():
            occurred = bool(streams.rarity.random() < prob)
            if occurred:
                rare_anim_events.append(event_name)
                if event_name == "golden-orbit-motion":
                    golden_orbit_shift = True
            trait_probs.append(TraitProbability(
                trait_name=f"anim_rare:{event_name}",
                value="enabled" if occurred else "absent",
                probability=prob if occurred else (1.0 - prob),
            ))

        animated_traits: list[str] = []
        if profile == "color-cycle":
            animated_traits.append("color_cycle")
        elif profile == "micro-orbit":
            animated_traits.extend(["center_real", "center_imag"])
        elif profile == "fractal-breath":
            animated_traits.append("zoom")
        elif profile == "orbit-trap-cycle":
            animated_traits.append("golden_orbit_phase")
        if golden_orbit_shift:
            animated_traits.append("golden_orbit_shift")

        trait_probs.append(TraitProbability(
            trait_name="anim_profile",
            value=profile,
            probability=1.0 / len(MANDELBROT_PROFILES),
        ))

        anim_params = MandelbrotAnimationParams(
            motion_profile=profile,
            color_cycles=color_cycles,
            orbit_radius=orbit_radius,
            zoom_pulse_amplitude=zoom_pulse_amplitude,
            golden_orbit_shift=golden_orbit_shift,
        )

        fps = options.fps
        frame_count = options.frame_count
        frame_duration_ms = frame_duration_ms_from_fps(fps)

        recipe = AnimationRecipe(
            animation_schema_version=ANIMATION_SCHEMA_VERSION,
            base_recipe=artwork_recipe,
            master_seed=artwork_recipe.seed,
            animation_seed=animation_seed,
            frame_count=frame_count,
            fps=fps,
            frame_duration_ms=frame_duration_ms,
            loop_count=options.loop_count,
            motion_profile=profile,
            motion_intensity=intensity,
            rotation_turns=1,
            color_cycles=color_cycles,
            pulse_count=1,
            direction="forward",
            animated_traits=tuple(animated_traits),
            rare_animation_events=tuple(rare_anim_events),
            retry_index=0,
            generator_animation_params=anim_params.to_dict(),
        )
        return recipe, trait_probs

    def render_frame(
        self,
        animation_recipe: AnimationRecipe,
        phase: float,
    ) -> UInt8Array:
        """Render a single mandelbrot-dream frame at *phase* ∈ [0.0, 1.0)."""
        base = animation_recipe.base_recipe
        ap = MandelbrotAnimationParams.from_dict(
            animation_recipe.generator_animation_params
        )
        p = MandelbrotDreamParams.from_dict(base.generator_params)

        width = base.width
        height = base.height
        profile = ap.motion_profile

        # ── Effective zoom (fractal-breath profile) ───────────────────────────
        if profile == "fractal-breath":
            zoom_mod = smooth_periodic_envelope(
                phase, cycles=1, bias=1.0,
                amplitude=ap.zoom_pulse_amplitude,
            )
            effective_zoom = p.zoom * zoom_mod
        else:
            effective_zoom = p.zoom

        # ── Camera centre (micro-orbit profile) ───────────────────────────────
        if profile == "micro-orbit":
            cx, cy = circular_orbit(
                phase,
                cx=p.center_real,
                cy=p.center_imag,
                radius=ap.orbit_radius,
            )
        else:
            cx, cy = p.center_real, p.center_imag

        # ── Colour cycle ──────────────────────────────────────────────────────
        color_shift = periodic_color_shift(phase, cycles=ap.color_cycles)
        effective_color_cycle = math.fmod(p.color_cycle + color_shift, 1.0)

        detail_cap = _DETAIL_MAX_ITER.get(base.detail_level, 256)
        max_iterations = min(p.max_iterations, detail_cap)

        aspect_ratio = width / height
        real_half = 1.6 / effective_zoom * aspect_ratio
        imag_half = 1.6 / effective_zoom

        real_axis = np.linspace(cx - real_half, cx + real_half, width)
        imag_axis = np.linspace(cy - imag_half, cy + imag_half, height)
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

        # ── Golden orbit (static or animated orbit-trap-cycle) ────────────────
        if "golden-orbit" in base.rare_events:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            if profile == "orbit-trap-cycle" or ap.golden_orbit_shift:
                # Phase-shift the target normalized time.
                trap_shift = 0.05 * cyclic_sine(phase, cycles=1)
            else:
                trap_shift = 0.0
            normalized_time = smooth_escape / max(max_iterations, 1)
            orbit_dist = np.abs(normalized_time - (1.0 / phi + trap_shift))
            orbit_ring = np.exp(-(orbit_dist ** 2) / 0.004)
            normalized = np.clip(normalized + 0.35 * orbit_ring * escaped_mask, 0.0, 1.0)

        if "perfect-alignment" in base.rare_events:
            if "perfect-loop" in animation_recipe.rare_animation_events:
                # Rotate the alignment pattern.
                rot_offset = math.tau * phase
                sym_pattern = np.abs(np.sin(
                    complex_plane.real * 8.0 * math.cos(rot_offset)
                    + complex_plane.imag * 8.0 * math.sin(rot_offset)
                )) * 0.15
            else:
                sym_pattern = np.abs(np.sin(complex_plane.real * 8.0)) * 0.15
            normalized = np.clip(normalized + sym_pattern * escaped_mask, 0.0, 1.0)

        palette = _PALETTE_REGISTRY.get(base.palette_name)
        palette_t = np.mod(effective_color_cycle + 0.85 * normalized, 1.0)

        a_arr = np.asarray(palette.cosine_a, dtype=np.float64)
        b_arr = np.asarray(palette.cosine_b, dtype=np.float64)
        c_arr = np.asarray(palette.cosine_c, dtype=np.float64)
        d_arr = np.asarray(palette.cosine_d, dtype=np.float64)
        t = palette_t[..., np.newaxis]
        rgb_float = a_arr + b_arr * np.cos(math.tau * (t * c_arr + d_arr))

        if p.interior_mode == "black":
            interior_rgb = np.array([0.04, 0.04, 0.04], dtype=np.float64)
        elif p.interior_mode == "white":
            interior_rgb = np.array([0.95, 0.95, 0.95], dtype=np.float64)
        elif p.interior_mode == "nebula":
            interior_rgb = np.asarray(palette.shadow_color, dtype=np.float64) * 1.2
        else:
            interior_rgb = np.asarray(palette.background_color, dtype=np.float64)

        interior_mask = active[..., np.newaxis]
        rgb_float = np.where(interior_mask, interior_rgb, rgb_float)

        brightness = np.where(active, 0.0, 0.18 + 0.82 * normalized ** 0.65)
        rgb_float = rgb_float * np.clip(brightness, 0.1, 1.0)[..., np.newaxis]

        h, w = height, width
        yi = (np.arange(h) - h / 2.0) / max(min(h, w) / 2.0, 0.5)
        xi = (np.arange(w) - w / 2.0) / max(min(h, w) / 2.0, 0.5)
        xx, yy = np.meshgrid(xi, yi)
        radius = np.hypot(xx, yy)

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=radius,
            x=xx,
            y=yy,
            lighting_mode=base.lighting_mode,
            background_mode=base.background_mode,
            accent_mode=base.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)
