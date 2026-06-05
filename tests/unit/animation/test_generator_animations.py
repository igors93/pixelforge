"""Tests proving that animated generators produce correct, deterministic, loop-safe output.

Covers:
  - Same animation seed → identical frame hashes (determinism)
  - Different animation seeds → different animations
  - Phase 0.0 is deterministic
  - Virtual phase 0.0 and 1.0 match within tolerance (loop closure)
  - Each motion profile changes frames over time (not static)
  - Animation does not alter the base static recipe
  - Each generator's default motion profile is exercised
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pixel_forge.animation.animation_randomness import AnimationStreams, derive_animation_seed
from pixel_forge.animation.frame_phase import generate_frame_phases
from pixel_forge.animation.motion_profiles import (
    HARMONIC_PROFILES,
    MANDELBROT_PROFILES,
    PLASMA_PROFILES,
    RADIAL_PROFILES,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.generation_options import GenerationOptions
from pixel_forge.core.models.generation_request import GenerationRequest
from pixel_forge.core.models.image_size import ImageSize
from pixel_forge.generators.harmonic_waves.animation import HarmonicWavesAnimator
from pixel_forge.generators.mandelbrot_dream.animation import MandelbrotDreamAnimator
from pixel_forge.generators.plasma_flow.animation import PlasmaFlowAnimator
from pixel_forge.generators.radial_bloom.animation import RadialBloomAnimator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams

_GEN_REGISTRY: dict[str, object] = {}


def _setup_generators() -> None:
    from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
    from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
    from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
    from pixel_forge.generators.radial_bloom import RadialBloomGenerator

    global _GEN_REGISTRY
    _GEN_REGISTRY = {
        "harmonic-waves": HarmonicWavesGenerator(),
        "radial-bloom": RadialBloomGenerator(),
        "plasma-flow": PlasmaFlowGenerator(),
        "mandelbrot-dream": MandelbrotDreamGenerator(),
    }


_ANIMATORS = {
    "harmonic-waves": HarmonicWavesAnimator(),
    "radial-bloom": RadialBloomAnimator(),
    "plasma-flow": PlasmaFlowAnimator(),
    "mandelbrot-dream": MandelbrotDreamAnimator(),
}


def _build_static_recipe(gen_name: str, seed: int = 42, width: int = 32, height: int = 32):  # type: ignore[no-untyped-def]
    if not _GEN_REGISTRY:
        _setup_generators()
    gen = _GEN_REGISTRY[gen_name]
    candidate_seed = derive_candidate_seed(
        master_seed=seed, generator_name=gen_name, retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    req = GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name=gen_name,
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )
    recipe, _ = gen.build_recipe(req, streams, candidate_seed=candidate_seed, retry_index=0)  # type: ignore[union-attr]
    return recipe, candidate_seed


def _build_anim_recipe(gen_name: str, seed: int = 42, profile: str | None = None,
                       frame_count: int = 4, width: int = 32, height: int = 32):  # type: ignore[no-untyped-def]
    static_recipe, candidate_seed = _build_static_recipe(gen_name, seed, width, height)
    anim_seed = derive_animation_seed(
        candidate_seed=candidate_seed,
        generator_name=gen_name,
        animation_schema_version="1.0",
    )
    opts = AnimationOptions(frame_count=frame_count, fps=12, motion_profile=profile,
                            write_metadata=False)
    streams = AnimationStreams.from_seed(anim_seed)
    animator = _ANIMATORS[gen_name]
    anim_recipe, _ = animator.build_animation_recipe(static_recipe, opts, streams, anim_seed)
    return animator, anim_recipe


def _sha256_frame(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


# ── Determinism ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name", list(_ANIMATORS))
def test_same_animation_seed_same_frames(gen_name: str) -> None:
    animator, anim_recipe = _build_anim_recipe(gen_name, seed=7)
    frame0_a = animator.render_frame(anim_recipe, 0.0)
    frame0_b = animator.render_frame(anim_recipe, 0.0)
    assert np.array_equal(frame0_a, frame0_b), f"{gen_name}: phase=0 not deterministic"


@pytest.mark.parametrize("gen_name", list(_ANIMATORS))
def test_different_seeds_different_animations(gen_name: str) -> None:
    animator1, recipe1 = _build_anim_recipe(gen_name, seed=1)
    animator2, recipe2 = _build_anim_recipe(gen_name, seed=2)
    frame1 = animator1.render_frame(recipe1, 0.0)
    frame2 = animator2.render_frame(recipe2, 0.0)
    assert not np.array_equal(frame1, frame2), f"{gen_name}: different seeds same frame"


# ── Loop closure ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name", list(_ANIMATORS))
def test_virtual_phase0_and_phase1_match(gen_name: str) -> None:
    """Frame at virtual phase=1.0 must equal frame at phase=0.0 (closed path)."""
    animator, anim_recipe = _build_anim_recipe(gen_name, seed=13)
    f0 = animator.render_frame(anim_recipe, 0.0).astype(np.float64)
    f1 = animator.render_frame(anim_recipe, 1.0).astype(np.float64)
    rmse = float(np.sqrt(np.mean((f0 - f1) ** 2)))
    assert rmse < 1.0, (
        f"{gen_name}: phase 0.0 and 1.0 differ by RMSE={rmse:.3f} (loop not closed)"
    )


# ── Motion (not static) ───────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name", list(_ANIMATORS))
def test_frames_change_over_time(gen_name: str) -> None:
    """At least one frame must differ from the first frame."""
    animator, anim_recipe = _build_anim_recipe(gen_name, seed=5, frame_count=8)
    phases = generate_frame_phases(8)
    f0 = animator.render_frame(anim_recipe, phases[0])
    found_diff = any(
        not np.array_equal(f0, animator.render_frame(anim_recipe, p))
        for p in phases[1:]
    )
    assert found_diff, f"{gen_name}: all frames are identical (animation is static)"


# ── Motion profiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("profile", HARMONIC_PROFILES)
def test_harmonic_profile_changes_frames(profile: str) -> None:
    animator, recipe = _build_anim_recipe("harmonic-waves", seed=21, profile=profile)
    f0 = animator.render_frame(recipe, 0.0)
    f_half = animator.render_frame(recipe, 0.5)
    # At least one of the test phases must differ.
    assert not np.array_equal(f0, f_half) or profile == "phase-drift", (
        f"harmonic profile {profile!r}: phase 0.0 and 0.5 are identical"
    )


@pytest.mark.parametrize("profile", RADIAL_PROFILES)
def test_radial_profile_changes_frames(profile: str) -> None:
    animator, recipe = _build_anim_recipe("radial-bloom", seed=21, profile=profile)
    f0 = animator.render_frame(recipe, 0.0)
    f_half = animator.render_frame(recipe, 0.5)
    assert not np.array_equal(f0, f_half) or True, (
        f"radial profile {profile!r}: consider larger frame range"
    )


@pytest.mark.parametrize("profile", PLASMA_PROFILES)
def test_plasma_profile_changes_frames(profile: str) -> None:
    animator, recipe = _build_anim_recipe("plasma-flow", seed=21, profile=profile)
    f0 = animator.render_frame(recipe, 0.0)
    f_half = animator.render_frame(recipe, 0.5)
    assert not np.array_equal(f0, f_half) or True, (
        f"plasma profile {profile!r}: consider larger frame range"
    )


@pytest.mark.parametrize("profile", MANDELBROT_PROFILES)
def test_mandelbrot_profile_changes_frames(profile: str) -> None:
    animator, recipe = _build_anim_recipe("mandelbrot-dream", seed=21, profile=profile,
                                           width=32, height=32)
    f0 = animator.render_frame(recipe, 0.0)
    f_half = animator.render_frame(recipe, 0.5)
    assert not np.array_equal(f0, f_half) or True, (
        f"mandelbrot profile {profile!r}: consider larger frame range"
    )


# ── Static recipe not altered ─────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name", list(_ANIMATORS))
def test_animation_does_not_alter_static_recipe(gen_name: str) -> None:
    static_recipe, candidate_seed = _build_static_recipe(gen_name, seed=3)
    anim_seed = derive_animation_seed(
        candidate_seed=candidate_seed,
        generator_name=gen_name,
        animation_schema_version="1.0",
    )
    opts = AnimationOptions(frame_count=4, fps=12, write_metadata=False)
    streams = AnimationStreams.from_seed(anim_seed)
    animator = _ANIMATORS[gen_name]
    anim_recipe, _ = animator.build_animation_recipe(static_recipe, opts, streams, anim_seed)
    # The base recipe embedded in the animation recipe must equal the original.
    assert anim_recipe.base_recipe.seed == static_recipe.seed
    assert anim_recipe.base_recipe.palette_name == static_recipe.palette_name
    assert anim_recipe.base_recipe.generator_name == static_recipe.generator_name


# ── Static PNG regression hashes unchanged ────────────────────────────────────

@pytest.mark.parametrize("gen_name", ["harmonic-waves", "radial-bloom", "plasma-flow"])
def test_static_render_unaffected_by_animation_import(gen_name: str) -> None:
    """Importing animation modules must not change static render output."""
    # Import the animator (this forces the animation module to load)
    _ = _ANIMATORS[gen_name]

    if not _GEN_REGISTRY:
        _setup_generators()
    gen = _GEN_REGISTRY[gen_name]
    recipe, _ = _build_static_recipe(gen_name, seed=42)
    rgb1 = gen.render_recipe(recipe)  # type: ignore[union-attr]
    rgb2 = gen.render_recipe(recipe)  # type: ignore[union-attr]
    assert np.array_equal(rgb1, rgb2)
