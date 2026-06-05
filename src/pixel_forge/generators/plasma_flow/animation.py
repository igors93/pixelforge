"""Animated renderer for the Plasma Flow generator.

Motion profiles:
  flow-cycle     – domain warp phases advance by integer cycles (default)
  vortex-orbit   – vortex positions orbit in circles
  filament-breath – filament angular frequency pulses
  plasma-tide    – flow direction bias sweeps through one full cycle

Rare animation events:
  reverse-flow   – the entire flow direction reverses mid-cycle
  eclipse-pulse  – singularity zone pulses in time with the flow
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
)
from pixel_forge.animation.motion_profiles import (
    PLASMA_DEFAULT_PROFILE,
    PLASMA_PROFILES,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION, AnimationRecipe
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.rendering import (
    apply_recipe_post_processing,
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.plasma_flow.generator import (
    _COMPLEXITY_WARP_CAP,
    _PALETTE_REGISTRY,
)
from pixel_forge.generators.plasma_flow.parameters import PlasmaFlowParams, VortexEntry
from pixel_forge.rarity.trait_probability import TraitProbability

_RARE_ANIMATION_EVENTS: dict[str, float] = {
    "reverse-flow": 1 / 12,
    "eclipse-pulse": 1 / 8,
}


@dataclass(frozen=True, slots=True)
class PlasmaFlowAnimationParams:
    """Generator-specific animation parameters for plasma-flow."""

    motion_profile: str
    flow_cycles: int         # integer warp phase advances per loop
    color_cycles: int        # integer palette cycles
    pulse_count: int         # integer pulses for breathing profiles
    vortex_orbit_radius: float  # orbit radius for vortex-orbit profile
    reverse_flow: bool       # rare: adds a reversed-direction overlay

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_profile": self.motion_profile,
            "flow_cycles": self.flow_cycles,
            "color_cycles": self.color_cycles,
            "pulse_count": self.pulse_count,
            "vortex_orbit_radius": self.vortex_orbit_radius,
            "reverse_flow": self.reverse_flow,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PlasmaFlowAnimationParams:
        return cls(
            motion_profile=str(d["motion_profile"]),
            flow_cycles=int(d["flow_cycles"]),
            color_cycles=int(d["color_cycles"]),
            pulse_count=int(d["pulse_count"]),
            vortex_orbit_radius=float(d["vortex_orbit_radius"]),
            reverse_flow=bool(d["reverse_flow"]),
        )


class PlasmaFlowAnimator:
    """Animated renderer for plasma-flow recipes."""

    @property
    def name(self) -> str:
        return "plasma-flow"

    def build_animation_recipe(
        self,
        artwork_recipe: ArtworkRecipe,
        options: AnimationOptions,
        streams: AnimationStreams,
        animation_seed: int,
    ) -> tuple[AnimationRecipe, list[TraitProbability]]:
        trait_probs: list[TraitProbability] = []

        profile = options.motion_profile
        if profile is None or profile not in PLASMA_PROFILES:
            profile = PLASMA_DEFAULT_PROFILE

        intensity = options.motion_intensity
        if intensity is None:
            intensity = float(streams.motion.uniform(0.5, 1.0))

        flow_cycles = int(streams.motion.integers(1, 4))
        color_cycles = int(streams.color.integers(1, 4))
        pulse_count = int(streams.motion.integers(1, 3))
        vortex_orbit_radius = float(streams.camera.uniform(0.05, 0.15))

        rare_anim_events: list[str] = []
        reverse_flow = False
        for event_name, prob in _RARE_ANIMATION_EVENTS.items():
            occurred = bool(streams.rarity.random() < prob)
            if occurred:
                rare_anim_events.append(event_name)
                if event_name == "reverse-flow":
                    reverse_flow = True
            trait_probs.append(TraitProbability(
                trait_name=f"anim_rare:{event_name}",
                value="enabled" if occurred else "absent",
                probability=prob if occurred else (1.0 - prob),
            ))

        animated_traits: list[str] = []
        if profile == "flow-cycle":
            animated_traits.append("warp_phases")
        elif profile == "vortex-orbit":
            animated_traits.append("vortex_positions")
        elif profile == "filament-breath":
            animated_traits.append("filament_freq")
        elif profile == "plasma-tide":
            animated_traits.append("direction_bias")
        if reverse_flow:
            animated_traits.append("reverse_overlay")

        trait_probs.append(TraitProbability(
            trait_name="anim_profile",
            value=profile,
            probability=1.0 / len(PLASMA_PROFILES),
        ))

        anim_params = PlasmaFlowAnimationParams(
            motion_profile=profile,
            flow_cycles=flow_cycles,
            color_cycles=color_cycles,
            pulse_count=pulse_count,
            vortex_orbit_radius=vortex_orbit_radius,
            reverse_flow=reverse_flow,
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
        """Render a single plasma-flow frame at *phase* ∈ [0.0, 1.0)."""
        from pixel_forge.core.models.image_size import ImageSize

        base = animation_recipe.base_recipe
        ap = PlasmaFlowAnimationParams.from_dict(
            animation_recipe.generator_animation_params
        )
        p = PlasmaFlowParams.from_dict(base.generator_params)

        size = ImageSize(width=base.width, height=base.height)
        field = build_coordinate_field(size)

        ph = list(p.phases) + [0.0] * 6

        # ── Warp phase drift (flow-cycle) ─────────────────────────────────────
        if ap.motion_profile == "flow-cycle":
            phase_delta = math.tau * ap.flow_cycles * phase
        else:
            phase_delta = 0.0

        # ── Animated vortex positions (vortex-orbit) ──────────────────────────
        if ap.motion_profile == "vortex-orbit" and p.vortex_data:
            animated_vortex_data: list[VortexEntry] = []
            for i, vd in enumerate(p.vortex_data):
                # Each vortex orbits with a different phase offset for variety.
                orbit_phase = math.fmod(phase + i / max(len(p.vortex_data), 1), 1.0)
                nx, ny = circular_orbit(
                    orbit_phase,
                    cx=vd.x,
                    cy=vd.y,
                    radius=ap.vortex_orbit_radius,
                )
                animated_vortex_data.append(VortexEntry(
                    x=nx, y=ny, strength=vd.strength, sign=vd.sign
                ))
        else:
            animated_vortex_data = list(p.vortex_data)

        # ── Plasma-tide: sweep direction bias ─────────────────────────────────
        if ap.motion_profile == "plasma-tide":
            # direction_bias sweeps through [0, 2π) → closes exactly at phase=1.
            tide_bias = math.tau * phase
        else:
            direction_bias_map = {
                "radial": 0.0,
                "diagonal": 0.5,
                "horizontal": 1.0,
                "turbulent": 1.5,
            }
            tide_bias = direction_bias_map.get(p.flow_direction, 0.0)

        shifted_radius = np.hypot(
            field.x_centered - p.center_x,
            field.y_centered - p.center_y,
        )

        plasma = (
            np.sin(field.x_unit * p.freq_low * math.pi + ph[0] + phase_delta + tide_bias)
            + np.sin(field.y_unit * p.freq_low * math.pi + ph[1] + phase_delta + tide_bias * 0.7)
            + np.sin(
                (field.x_unit + field.y_unit) * p.freq_high * math.pi
                + ph[2] + phase_delta + tide_bias
            )
            + np.sin(shifted_radius * p.freq_high * 0.85 + ph[3])
        ) / 4.0

        warp_cap = _COMPLEXITY_WARP_CAP.get(base.complexity_level, 2)
        effective_warp = min(p.warp_stages, warp_cap)

        wx, wy = field.x_centered, field.y_centered
        if effective_warp >= 1:
            warp_y = p.warp_strength * np.sin(
                field.y_unit * p.freq_low * math.pi + ph[4] + phase_delta
            )
            warp_x = p.warp_strength * np.cos(
                field.x_unit * p.freq_low * math.pi + ph[5] + phase_delta
            )
            wx = field.x_centered + warp_y
            wy = field.y_centered + warp_x

        for vd in animated_vortex_data:
            dx = wx - vd.x
            dy = wy - vd.y
            r2 = dx * dx + dy * dy + 0.01
            curl_x = -vd.sign * dy / r2 * vd.strength
            curl_y = vd.sign * dx / r2 * vd.strength
            vortex_r = np.hypot(
                wx + curl_x - (field.x_centered - p.center_x),
                wy + curl_y - (field.y_centered - p.center_y),
            )
            vortex_contribution = np.sin(vortex_r * p.freq_low * 2.0 + ph[0] + phase_delta)
            plasma = plasma + 0.15 * vortex_contribution / max(len(animated_vortex_data), 1)

        if effective_warp >= 2 and p.turbulence > 0:
            plasma = plasma + p.turbulence * np.sin(
                wx * p.freq_high * math.pi * 0.5 + wy * p.freq_high * math.pi * 0.5 + ph[2]
            )

        # ── Filament-surge (static from recipe, animated by filament-breath) ──
        if "filament-surge" in base.rare_events:
            angle_field = np.arctan2(field.y_centered, field.x_centered)
            if ap.motion_profile == "filament-breath":
                # Filament frequency pulses over the loop.
                freq_mod = 1.0 + 0.4 * cyclic_sine(phase, cycles=ap.pulse_count)
                fil_freq = p.freq_high * 0.4 * freq_mod
            else:
                fil_freq = p.freq_high * 0.4
            filament = np.sin(angle_field * 12.0 + shifted_radius * fil_freq) ** 8
            plasma = plasma + 0.20 * filament

        # Reverse-flow rare animation event: overlay inverted phase.
        if ap.reverse_flow:
            reverse_strength = 0.15 * abs(cyclic_sine(phase, cycles=ap.pulse_count))
            plasma = plasma + reverse_strength * (
                np.sin(field.x_unit * p.freq_low * math.pi + ph[0] - phase_delta)
                + np.sin(field.y_unit * p.freq_low * math.pi + ph[1] - phase_delta)
            ) / 2.0

        palette_position = np.clip(0.5 + 0.5 * plasma + 0.08 * shifted_radius, 0.0, 1.0)
        brightness = np.clip(
            0.46 + 0.54 * (0.5 + 0.5 * np.cos(plasma * math.pi + shifted_radius * 1.5)),
            0.0, 1.0,
        )

        if "singularity" in base.rare_events:
            singularity_mask = np.exp(-(shifted_radius ** 2) / 0.008)
            brightness = np.clip(brightness * (1.0 - 0.9 * singularity_mask), 0.0, 1.0)

        palette = _PALETTE_REGISTRY.get(base.palette_name)
        color_shift = periodic_color_shift(phase, cycles=ap.color_cycles)
        d_arr = np.asarray([
            palette.cosine_d[0] + p.palette_phase_shift + color_shift,
            palette.cosine_d[1] + p.palette_phase_shift + color_shift,
            palette.cosine_d[2] + p.palette_phase_shift + color_shift,
        ], dtype=np.float64)
        a_arr = np.asarray(palette.cosine_a, dtype=np.float64)
        b_arr = np.asarray(palette.cosine_b, dtype=np.float64)
        c_arr = np.asarray(palette.cosine_c, dtype=np.float64)
        t = palette_position[..., np.newaxis]
        rgb_float = a_arr + b_arr * np.cos(math.tau * (t * c_arr + d_arr))
        rgb_float = rgb_float * np.clip(brightness, 0.0, 1.0)[..., np.newaxis]

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=shifted_radius,
            x=field.x_centered,
            y=field.y_centered,
            lighting_mode=base.lighting_mode,
            background_mode=base.background_mode,
            accent_mode=base.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)
