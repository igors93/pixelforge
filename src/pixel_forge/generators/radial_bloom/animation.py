"""Animated renderer for the Radial Bloom generator.

Motion profiles:
  bloom-pulse      – petal glow envelope breathes in and out (default)
  radial-rotation  – entire field rotates continuously
  orbital-bloom    – crown ring radii orbit outward and back
  spiral-breath    – ripple phase advances smoothly
  eclipse-pulse    – orbital halo ring radius oscillates

Rare animation events:
  counter-rotating-halo – secondary halo rotates opposite to main field
  synchronized-bloom    – petal count pulses in sync with rotation
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pixel_forge.animation.animation_randomness import AnimationStreams
from pixel_forge.animation.loop_math import (
    cyclic_cosine,
    cyclic_sine,
    periodic_color_shift,
    periodic_rotation,
    smooth_periodic_envelope,
)
from pixel_forge.animation.motion_profiles import (
    RADIAL_DEFAULT_PROFILE,
    RADIAL_PROFILES,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION, AnimationRecipe
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.core.models.artwork_traits import ComplexityLevel, DetailLevel, SymmetryMode
from pixel_forge.generators.common.fields import build_coordinate_field
from pixel_forge.generators.common.rendering import (
    apply_recipe_post_processing,
    apply_symmetry_to_coordinates,
    build_rgb_float_from_palette,
    rgb_float_to_bytes,
)
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.radial_bloom.generator import _PALETTE_REGISTRY
from pixel_forge.generators.radial_bloom.parameters import RadialBloomParams
from pixel_forge.rarity.trait_probability import TraitProbability

_RARE_ANIMATION_EVENTS: dict[str, float] = {
    "counter-rotating-halo": 1 / 12,
    "synchronized-bloom": 1 / 8,
}


@dataclass(frozen=True, slots=True)
class RadialBloomAnimationParams:
    """Generator-specific animation parameters for radial-bloom."""

    motion_profile: str
    rotation_turns: int      # integer full turns per loop
    color_cycles: int        # integer palette cycles
    pulse_count: int         # integer breathing pulses
    ripple_drift_cycles: int # integer ripple phase advances
    counter_halo: bool       # rare: counter-rotating halo ring

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_profile": self.motion_profile,
            "rotation_turns": self.rotation_turns,
            "color_cycles": self.color_cycles,
            "pulse_count": self.pulse_count,
            "ripple_drift_cycles": self.ripple_drift_cycles,
            "counter_halo": self.counter_halo,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RadialBloomAnimationParams:
        return cls(
            motion_profile=str(d["motion_profile"]),
            rotation_turns=int(d["rotation_turns"]),
            color_cycles=int(d["color_cycles"]),
            pulse_count=int(d["pulse_count"]),
            ripple_drift_cycles=int(d["ripple_drift_cycles"]),
            counter_halo=bool(d["counter_halo"]),
        )


class RadialBloomAnimator:
    """Animated renderer for radial-bloom recipes."""

    @property
    def name(self) -> str:
        return "radial-bloom"

    def build_animation_recipe(
        self,
        artwork_recipe: ArtworkRecipe,
        options: AnimationOptions,
        streams: AnimationStreams,
        animation_seed: int,
    ) -> tuple[AnimationRecipe, list[TraitProbability]]:
        trait_probs: list[TraitProbability] = []

        profile = options.motion_profile
        if profile is None or profile not in RADIAL_PROFILES:
            profile = RADIAL_DEFAULT_PROFILE

        intensity = options.motion_intensity
        if intensity is None:
            intensity = float(streams.motion.uniform(0.5, 1.0))

        rotation_turns = int(streams.motion.integers(1, 3))
        color_cycles = int(streams.color.integers(1, 4))
        pulse_count = int(streams.motion.integers(1, 3))
        ripple_drift_cycles = int(streams.motion.integers(1, 4))

        rare_anim_events: list[str] = []
        counter_halo = False
        for event_name, prob in _RARE_ANIMATION_EVENTS.items():
            occurred = bool(streams.rarity.random() < prob)
            if occurred:
                rare_anim_events.append(event_name)
                if event_name == "counter-rotating-halo":
                    counter_halo = True
            trait_probs.append(TraitProbability(
                trait_name=f"anim_rare:{event_name}",
                value="enabled" if occurred else "absent",
                probability=prob if occurred else (1.0 - prob),
            ))

        animated_traits: list[str] = []
        if profile == "bloom-pulse":
            animated_traits.append("glow_spread")
        elif profile == "radial-rotation":
            animated_traits.append("rotation")
        elif profile == "orbital-bloom":
            animated_traits.append("crown_radius")
        elif profile == "spiral-breath":
            animated_traits.append("ripple_phase")
        elif profile == "eclipse-pulse":
            animated_traits.append("halo_radius")
        if counter_halo:
            animated_traits.append("counter_halo_rotation")

        trait_probs.append(TraitProbability(
            trait_name="anim_profile",
            value=profile,
            probability=1.0 / len(RADIAL_PROFILES),
        ))

        anim_params = RadialBloomAnimationParams(
            motion_profile=profile,
            rotation_turns=rotation_turns,
            color_cycles=color_cycles,
            pulse_count=pulse_count,
            ripple_drift_cycles=ripple_drift_cycles,
            counter_halo=counter_halo,
        )

        fps = options.fps
        frame_count = options.frame_count
        frame_duration_ms = round(1000 / fps)

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
        """Render a single radial-bloom frame at *phase* ∈ [0.0, 1.0)."""
        from pixel_forge.core.models.image_size import ImageSize

        base = animation_recipe.base_recipe
        ap = RadialBloomAnimationParams.from_dict(
            animation_recipe.generator_animation_params
        )
        p = RadialBloomParams.from_dict(base.generator_params)

        size = ImageSize(width=base.width, height=base.height)
        field = build_coordinate_field(size)

        xc, yc = apply_symmetry_to_coordinates(
            field.x_centered, field.y_centered, base.symmetry_mode
        )

        profile = ap.motion_profile

        # ── Rotation angle ───────────────────────────────────────────────────
        if profile == "radial-rotation":
            rotation_offset = periodic_rotation(phase, turns=ap.rotation_turns)
        else:
            rotation_offset = 0.0

        angle = np.arctan2(yc, xc) + rotation_offset
        radius = np.hypot(xc, yc)

        if base.symmetry_mode == SymmetryMode.BROKEN or "broken-symmetry" in base.rare_events:
            angle = np.arctan2(field.y_centered - 0.12, field.x_centered + 0.08)
            radius = np.hypot(field.x_centered + 0.08, field.y_centered - 0.12)
            angle = angle + rotation_offset

        # Phyllotaxis distortion (static from recipe).
        if p.phyllotaxis:
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))
            angle = angle + radius * golden_angle * (2.0 if p.spiral_clockwise else -2.0)

        # ── Breathing glow spread (bloom-pulse) ──────────────────────────────
        if profile == "bloom-pulse":
            amplitude = 0.12 * animation_recipe.motion_intensity
            effective_glow = p.glow_spread * (
                1.0 + amplitude * cyclic_sine(phase, cycles=ap.pulse_count)
            )
        else:
            effective_glow = p.glow_spread

        # ── Ripple phase drift (spiral-breath) ───────────────────────────────
        if profile == "spiral-breath":
            ripple_phase_offset = math.tau * ap.ripple_drift_cycles * phase
        else:
            ripple_phase_offset = 0.0

        petal_base = np.abs(np.sin(angle * p.primary_petals + p.phase))
        petal_wave = petal_base ** p.petal_sharpness
        petal_envelope = np.exp(
            -(radius ** 2) / (effective_glow * p.petal_radial_scale + 1e-9)
        )

        ribbon_wave = np.cos(
            angle * p.secondary_petals
            - radius * p.ripple_frequency * 0.75
            - p.phase * 0.5
            + ripple_phase_offset
        )

        effective_ripple_count = p.radial_ripple_count
        if base.detail_level == DetailLevel.LOW:
            effective_ripple_count = min(effective_ripple_count, 1)

        ripple = np.zeros_like(radius)
        for i in range(effective_ripple_count):
            freq_mult = 1.0 + i * 0.5
            ripple = ripple + np.sin(
                radius * p.ripple_frequency * freq_mult
                + (p.phase + ripple_phase_offset) * (i + 1)
            )
        ripple = ripple / max(effective_ripple_count, 1)
        fine_ripple = 0.5 + 0.5 * ripple

        center_glow = np.exp(-(radius ** 2) / (effective_glow + 1e-9))

        # ── Crown rings (orbital-bloom profile orbits them) ──────────────────
        has_triple_crown = "triple-crown" in base.rare_events
        complexity_crown_mult = {
            ComplexityLevel.MINIMAL: 0,
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 1,
            ComplexityLevel.COMPLEX: 1,
            ComplexityLevel.INTRICATE: 2,
        }.get(base.complexity_level, 1)
        if has_triple_crown:
            actual_crowns = max(p.crown_count, 3) * max(complexity_crown_mult, 1)
        else:
            actual_crowns = p.crown_count * complexity_crown_mult

        crown_contribution = np.zeros_like(radius)
        for i in range(actual_crowns):
            base_crown_r = 0.35 + i * 0.25
            if profile == "orbital-bloom":
                # Crown ring orbits slightly outward and back.
                orbit_shift = 0.05 * cyclic_sine(phase, cycles=ap.pulse_count)
                crown_r = base_crown_r + orbit_shift
            else:
                crown_r = base_crown_r
            crown_contribution = crown_contribution + np.exp(
                -((radius - crown_r) ** 2) / 0.004
            )
        crown_contribution = np.clip(crown_contribution, 0.0, 1.0)

        # ── Orbital halo and eclipse pulse ───────────────────────────────────
        halo = np.zeros_like(radius)
        if "orbital-halo" in base.rare_events:
            if profile == "eclipse-pulse":
                halo_r = 0.72 + 0.08 * cyclic_sine(phase, cycles=ap.pulse_count)
            else:
                halo_r = 0.72
            halo = np.exp(-((radius - halo_r) ** 2) / 0.003)

        # Counter-rotating halo (rare animation event).
        if ap.counter_halo:
            counter_angle = angle - 2.0 * periodic_rotation(phase, turns=ap.rotation_turns)
            halo_ring_r = 0.85
            counter_halo_field = np.exp(-((radius - halo_ring_r) ** 2) / 0.006) * \
                (0.5 + 0.5 * np.sin(counter_angle * 3.0))
            halo = np.clip(halo + 0.4 * counter_halo_field, 0.0, 1.0)

        # Static rare event fields.
        spiral_field = np.zeros_like(radius)
        if "golden-spiral" in base.rare_events:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            spiral_field = np.sin(radius * phi * 12.0 + angle * 3.0 + p.phase) * 0.5 + 0.5

        center_value_mod = np.ones_like(radius)
        if "black-hole-center" in base.rare_events:
            center_value_mod = 1.0 - np.exp(-(radius ** 2) / 0.015)

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

        if p.center_mode == "void":
            brightness = brightness * center_value_mod
        elif p.center_mode == "bright":
            brightness = np.clip(brightness + 0.25 * center_glow, 0.0, 1.0)
        elif p.center_mode == "dark-star":
            brightness = brightness * (1.0 - 0.5 * center_glow)

        palette = _PALETTE_REGISTRY.get(base.palette_name)
        rgb_float = build_rgb_float_from_palette(palette_position, palette, brightness)

        rgb_float = apply_recipe_post_processing(
            rgb_float,
            radius=radius,
            x=xc,
            y=yc,
            lighting_mode=base.lighting_mode,
            background_mode=base.background_mode,
            accent_mode=base.accent_mode,
            palette=palette,
        )

        return rgb_float_to_bytes(rgb_float)
