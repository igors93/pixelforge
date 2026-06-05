"""Animated renderer for the Harmonic Waves generator.

Each motion profile applies a different closed periodic transformation to the
static recipe parameters. All transformations use integer cycle counts so
that the animated path closes exactly at phase=1.0.

Motion profiles:
  phase-drift    – wave phases advance by integer cycles per loop (default)
  rotating-field – the coordinate system rotates by integer full turns
  breathing-waves – freq_scale oscillates sinusoidally (integer pulses)
  dual-harmonic  – two frequency layers drift at different integer speeds
  color-orbit    – palette phase cycles by integer full cycles

Rare animation events:
  dual-rotation   – adds a counter-rotating secondary layer
  luminous-pulse  – luminous halo ring beats in time with the motion
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pixel_forge.animation.animation_randomness import AnimationStreams
from pixel_forge.animation.frame_timing import frame_duration_ms_from_fps
from pixel_forge.animation.loop_math import (
    cyclic_sine,
    periodic_color_shift,
    periodic_rotation,
    smooth_periodic_envelope,
)
from pixel_forge.animation.motion_profiles import (
    HARMONIC_DEFAULT_PROFILE,
    HARMONIC_PROFILES,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION, AnimationRecipe
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.core.models.artwork_traits import SymmetryMode
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.rendering import (
    apply_recipe_post_processing,
    apply_symmetry_to_coordinates,
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.harmonic_waves.generator import (
    _COMPLEXITY_LAYERS,
    _FREQUENCY_SETS,
    _PALETTE_REGISTRY,
)
from pixel_forge.generators.harmonic_waves.parameters import HarmonicWavesParams
from pixel_forge.rarity.trait_probability import TraitProbability

_RARE_ANIMATION_EVENTS: dict[str, float] = {
    "dual-rotation": 1 / 15,
    "luminous-pulse": 1 / 10,
}


@dataclass(frozen=True, slots=True)
class HarmonicWavesAnimationParams:
    """Generator-specific parameters for harmonic-waves animation."""

    motion_profile: str
    rotation_turns: int      # integer full turns per loop
    color_cycles: int        # integer palette cycles per loop
    pulse_count: int         # integer frequency pulses per loop
    phase_drift_cycles: int  # integer wave phase advances per loop
    warp_drift_cycles: int   # integer warp phase advances per loop
    dual_rotation: bool      # rare: add counter-rotating secondary field

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_profile": self.motion_profile,
            "rotation_turns": self.rotation_turns,
            "color_cycles": self.color_cycles,
            "pulse_count": self.pulse_count,
            "phase_drift_cycles": self.phase_drift_cycles,
            "warp_drift_cycles": self.warp_drift_cycles,
            "dual_rotation": self.dual_rotation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HarmonicWavesAnimationParams:
        return cls(
            motion_profile=str(d["motion_profile"]),
            rotation_turns=int(d["rotation_turns"]),
            color_cycles=int(d["color_cycles"]),
            pulse_count=int(d["pulse_count"]),
            phase_drift_cycles=int(d["phase_drift_cycles"]),
            warp_drift_cycles=int(d["warp_drift_cycles"]),
            dual_rotation=bool(d["dual_rotation"]),
        )


class HarmonicWavesAnimator:
    """Animated renderer for harmonic-waves recipes."""

    @property
    def name(self) -> str:
        return "harmonic-waves"

    def build_animation_recipe(
        self,
        artwork_recipe: ArtworkRecipe,
        options: AnimationOptions,
        streams: AnimationStreams,
        animation_seed: int,
    ) -> tuple[AnimationRecipe, list[TraitProbability]]:
        """Sample animation motion parameters and return the AnimationRecipe."""
        trait_probs: list[TraitProbability] = []

        # Motion profile
        profile = options.motion_profile
        if profile is None or profile not in HARMONIC_PROFILES:
            profile = HARMONIC_DEFAULT_PROFILE

        intensity = options.motion_intensity
        if intensity is None:
            intensity = float(streams.motion.uniform(0.5, 1.0))

        # Sample integer cycle counts — must be integers for closed paths.
        rotation_turns = int(streams.motion.integers(1, 3))  # 1 or 2 full turns
        color_cycles = int(streams.color.integers(1, 4))      # 1–3 palette cycles
        pulse_count = int(streams.motion.integers(1, 3))      # 1–2 freq pulses
        phase_drift_cycles = int(streams.motion.integers(2, 5))  # 2–4 phase advances
        warp_drift_cycles = int(streams.motion.integers(1, 3))

        # Rare animation events
        rare_anim_events: list[str] = []
        dual_rotation = False
        luminous_pulse = False
        for event_name, prob in _RARE_ANIMATION_EVENTS.items():
            occurred = bool(streams.rarity.random() < prob)
            if occurred:
                rare_anim_events.append(event_name)
                if event_name == "dual-rotation":
                    dual_rotation = True
                if event_name == "luminous-pulse":
                    luminous_pulse = True
            trait_probs.append(TraitProbability(
                trait_name=f"anim_rare:{event_name}",
                value="enabled" if occurred else "absent",
                probability=prob if occurred else (1.0 - prob),
            ))

        # Animated-trait tracking
        animated_traits: list[str] = []
        if profile == "phase-drift":
            animated_traits.append("wave_phases")
        elif profile == "rotating-field":
            animated_traits.append("rotation")
        elif profile == "breathing-waves":
            animated_traits.append("freq_scale")
        elif profile == "dual-harmonic":
            animated_traits.extend(["wave_phases", "freq_scale"])
        elif profile == "color-orbit":
            animated_traits.append("palette_phase")
        if dual_rotation:
            animated_traits.append("counter_rotation")
        if luminous_pulse:
            animated_traits.append("halo_pulse")

        # Record profile and direction as trait probabilities.
        trait_probs.append(TraitProbability(
            trait_name="anim_profile",
            value=profile,
            probability=1.0 / len(HARMONIC_PROFILES),
        ))

        anim_params = HarmonicWavesAnimationParams(
            motion_profile=profile,
            rotation_turns=rotation_turns,
            color_cycles=color_cycles,
            pulse_count=pulse_count,
            phase_drift_cycles=phase_drift_cycles,
            warp_drift_cycles=warp_drift_cycles,
            dual_rotation=dual_rotation,
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
            rotation_turns=rotation_turns,
            color_cycles=color_cycles,
            pulse_count=pulse_count,
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
        """Render a single harmonic-waves frame at *phase* ∈ [0.0, 1.0)."""
        # Phase 1.0 represents the same loop position as phase 0.0.
        phase = phase % 1.0

        from pixel_forge.core.models.image_size import ImageSize

        base = animation_recipe.base_recipe
        ap = HarmonicWavesAnimationParams.from_dict(
            animation_recipe.generator_animation_params
        )
        p = HarmonicWavesParams.from_dict(base.generator_params)

        size = ImageSize(width=base.width, height=base.height)
        field = build_coordinate_field(size)

        # Symmetry + broken symmetry coordinate transform.
        xc, yc = apply_symmetry_to_coordinates(
            field.x_centered, field.y_centered, base.symmetry_mode
        )
        if base.symmetry_mode == SymmetryMode.BROKEN or "broken-symmetry" in base.rare_events:
            xc = field.x_centered + 0.10
            yc = field.y_centered - 0.08

        # ── Profile-dependent coordinate rotation ────────────────────────────
        profile = ap.motion_profile
        base_rotation = p.rotation

        if profile == "rotating-field":
            delta_r = periodic_rotation(phase, turns=ap.rotation_turns)
            effective_rotation = base_rotation + delta_r
        else:
            effective_rotation = base_rotation

        cos_r = math.cos(effective_rotation)
        sin_r = math.sin(effective_rotation)
        rx = xc * cos_r - yc * sin_r
        ry = xc * sin_r + yc * cos_r

        # Dual-rotation rare event: overlay counter-rotating field.
        if ap.dual_rotation:
            delta_r2 = periodic_rotation(phase, turns=ap.rotation_turns)
            cos_r2 = math.cos(base_rotation - delta_r2)
            sin_r2 = math.sin(base_rotation - delta_r2)
            rx2 = xc * cos_r2 - yc * sin_r2
            ry2 = xc * sin_r2 + yc * cos_r2

        # ── Phase offsets ────────────────────────────────────────────────────
        # phase-drift: advance all wave phases by an integer number of cycles.
        if profile == "phase-drift" or profile == "dual-harmonic":
            delta_phase = math.tau * ap.phase_drift_cycles * phase
        else:
            delta_phase = 0.0

        # breathing-waves / dual-harmonic: modulate freq_scale.
        if profile == "breathing-waves" or profile == "dual-harmonic":
            freq_mod = smooth_periodic_envelope(
                phase,
                cycles=ap.pulse_count,
                bias=1.0,
                amplitude=0.25 * animation_recipe.motion_intensity,
            )
            effective_freq_scale = p.freq_scale * freq_mod
        else:
            effective_freq_scale = p.freq_scale

        # ── Domain warp ──────────────────────────────────────────────────────
        phases = list(p.phases)
        ph = phases + [0.0] * 8

        # Animate warp phase for "phase-drift" and related profiles.
        warp_delta = math.tau * ap.warp_drift_cycles * phase if profile in (
            "phase-drift", "dual-harmonic"
        ) else 0.0

        wx, wy = rx, ry
        if p.warp_stages >= 1:
            wx = rx + p.warp_strength * np.sin(ry * p.warp_freq_y + ph[0] + warp_delta)
            wy = ry + p.warp_strength * np.cos(rx * p.warp_freq_x + ph[1] + warp_delta)
        if p.warp_stages >= 2:
            s2 = p.warp_strength * 0.5
            wx = wx + s2 * np.cos(wy * p.warp_freq_y * 0.7 + ph[2] + warp_delta * 0.7)
            wy = wy + s2 * np.sin(wx * p.warp_freq_x * 0.7 + ph[3] + warp_delta * 0.7)

        r_warped = np.hypot(wx, wy)
        a_warped = np.arctan2(wy, wx)

        # Dual-rotation secondary warp.
        wx2: NDArray[np.float64] | None = None
        wy2: NDArray[np.float64] | None = None

        if ap.dual_rotation:
            wx2 = rx2
            wy2 = ry2

            if p.warp_stages >= 1:
                wx2 = rx2 + p.warp_strength * 0.5 * np.sin(
                    ry2 * p.warp_freq_y + ph[0] - warp_delta
                )
                wy2 = ry2 + p.warp_strength * 0.5 * np.cos(
                    rx2 * p.warp_freq_x + ph[1] - warp_delta
                )

        # ── Frequency set ────────────────────────────────────────────────────
        freq_tuple: tuple[float, ...] = (1.0, 1.618, 1.618 ** 2, 2.0)
        for name, fset in _FREQUENCY_SETS:
            if name == p.freq_set_name:
                freq_tuple = fset
                break

        max_layers = _COMPLEXITY_LAYERS.get(base.complexity_level, 6)
        active_layers = min(p.layer_count, max_layers, len(freq_tuple))

        combined = np.zeros_like(wx)
        freqs = list(freq_tuple)
        layer_weights = [0.40, 0.30, 0.20, 0.10, 0.08, 0.06]

        for i in range(active_layers):
            ph_i = ph[i % len(ph)] + delta_phase
            ph_j = ph[(i + 1) % len(ph)] + delta_phase
            f = freqs[i] * effective_freq_scale
            wave = np.sin(wx * f + 1.4 * np.sin(wy * (f * 0.55) + ph_j) + ph_i)
            w = layer_weights[i] if i < len(layer_weights) else 0.05
            combined = combined + w * wave

            # Secondary counter-rotating contribution.
            if ap.dual_rotation and wx2 is not None and wy2 is not None:
                wave2 = np.sin(wx2 * f + 1.4 * np.sin(wy2 * (f * 0.55) + ph_j) + ph_i)
                combined = combined + (w * 0.30) * wave2

        # Spiral.
        ph_sp = ph[-2] if len(ph) >= 2 else 0.0
        spiral_freq = effective_freq_scale * freqs[1] if len(freqs) > 1 else effective_freq_scale
        spiral = np.sin(r_warped * spiral_freq + a_warped * 3.0 + ph_sp + delta_phase)
        combined = combined + 0.12 * spiral

        palette_position = np.clip(0.5 + 0.5 * np.sin(combined * 2.0 + r_warped * 0.5), 0.0, 1.0)
        brightness = np.clip(
            0.52 + 0.48 * (0.5 + 0.5 * np.cos(combined * math.pi - spiral * 0.6)), 0.0, 1.0
        )

        # Luminous halo: static from recipe or pulsed by rare animation event.
        if "luminous-halo" in base.rare_events:
            if "luminous-pulse" in animation_recipe.rare_animation_events:
                pulse = (1.0 + cyclic_sine(phase, cycles=ap.pulse_count)) / 2.0
                halo_ring = np.exp(-((r_warped - 0.85) ** 2) / 0.005) * pulse
            else:
                halo_ring = np.exp(-((r_warped - 0.85) ** 2) / 0.005)
            brightness = np.clip(brightness + 0.5 * halo_ring, 0.0, 1.0)

        # ── Color orbit: advance palette phase ───────────────────────────────
        pd = p.palette_phase_d
        phase_d = (float(pd[0]), float(pd[1]) if len(pd) > 1 else 0.0,
                   float(pd[2]) if len(pd) > 2 else 0.0)

        if profile == "color-orbit":
            shift = periodic_color_shift(phase, cycles=ap.color_cycles)
            phase_d = (phase_d[0] + shift, phase_d[1] + shift, phase_d[2] + shift)

        palette = _PALETTE_REGISTRY.get(base.palette_name)
        a_arr = np.asarray(palette.cosine_a, dtype=np.float64)
        b_arr = np.asarray(palette.cosine_b, dtype=np.float64)
        c_arr = np.asarray(palette.cosine_c, dtype=np.float64)
        d_arr = np.asarray(phase_d, dtype=np.float64)
        t = palette_position[..., np.newaxis]
        rgb_float = a_arr + b_arr * np.cos(math.tau * (t * c_arr + d_arr))
        rgb_float = rgb_float * np.clip(brightness, 0.0, 1.0)[..., np.newaxis]

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=r_warped,
            x=xc,
            y=yc,
            lighting_mode=base.lighting_mode,
            background_mode=base.background_mode,
            accent_mode=base.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)
