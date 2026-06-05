"""Tests for deterministic animation seed derivation."""

from __future__ import annotations

from pixel_forge.animation.animation_randomness import (
    AnimationStreams,
    derive_animation_retry_seed,
    derive_animation_seed,
)


def test_derive_animation_seed_deterministic() -> None:
    s1 = derive_animation_seed(
        candidate_seed=12345,
        generator_name="harmonic-waves",
        animation_schema_version="1.0",
    )
    s2 = derive_animation_seed(
        candidate_seed=12345,
        generator_name="harmonic-waves",
        animation_schema_version="1.0",
    )
    assert s1 == s2


def test_derive_animation_seed_differs_for_different_candidate() -> None:
    s1 = derive_animation_seed(
        candidate_seed=1,
        generator_name="harmonic-waves",
        animation_schema_version="1.0",
    )
    s2 = derive_animation_seed(
        candidate_seed=2,
        generator_name="harmonic-waves",
        animation_schema_version="1.0",
    )
    assert s1 != s2


def test_derive_animation_seed_differs_from_static_seed() -> None:
    # Animation seed must not equal the candidate seed.
    candidate = 42
    anim = derive_animation_seed(
        candidate_seed=candidate,
        generator_name="harmonic-waves",
        animation_schema_version="1.0",
    )
    assert anim != candidate


def test_derive_animation_retry_seed_deterministic() -> None:
    base = derive_animation_seed(
        candidate_seed=99,
        generator_name="radial-bloom",
        animation_schema_version="1.0",
    )
    r1 = derive_animation_retry_seed(animation_seed=base, retry_index=1)
    r2 = derive_animation_retry_seed(animation_seed=base, retry_index=1)
    assert r1 == r2


def test_derive_animation_retry_seed_differs_by_index() -> None:
    base = derive_animation_seed(
        candidate_seed=99,
        generator_name="plasma-flow",
        animation_schema_version="1.0",
    )
    r0 = derive_animation_retry_seed(animation_seed=base, retry_index=0)
    r1 = derive_animation_retry_seed(animation_seed=base, retry_index=1)
    assert r0 != r1


def test_animation_streams_independent() -> None:
    """Sampling from one stream must not affect another."""
    seed = 777
    streams = AnimationStreams.from_seed(seed)
    v_motion = float(streams.motion.random())

    streams2 = AnimationStreams.from_seed(seed)
    _ = streams2.color.random()    # consume color stream first
    v_motion2 = float(streams2.motion.random())

    assert v_motion == v_motion2


def test_animation_streams_reproducible() -> None:
    s1 = AnimationStreams.from_seed(42)
    s2 = AnimationStreams.from_seed(42)
    assert float(s1.motion.random()) == float(s2.motion.random())
    assert float(s1.color.random()) == float(s2.color.random())
