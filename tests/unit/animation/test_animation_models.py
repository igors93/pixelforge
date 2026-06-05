"""Tests for animation domain models."""

from __future__ import annotations

from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION


def test_animation_options_defaults() -> None:
    opts = AnimationOptions()
    assert opts.frame_count == 24
    assert opts.fps == 24
    assert opts.loop_count == 0
    assert opts.motion_profile is None
    assert opts.gif_colors == 256
    assert opts.gif_dither == "none"
    assert opts.write_metadata is True
    assert opts.strict_temporal_quality is False


def test_animation_options_custom() -> None:
    opts = AnimationOptions(
        frame_count=48,
        fps=30,
        loop_count=3,
        motion_profile="radial-rotation",
        gif_colors=128,
        gif_dither="floyd-steinberg",
    )
    assert opts.frame_count == 48
    assert opts.fps == 30
    assert opts.loop_count == 3
    assert opts.motion_profile == "radial-rotation"
    assert opts.gif_colors == 128
    assert opts.gif_dither == "floyd-steinberg"


def test_animation_schema_version_defined() -> None:
    assert isinstance(ANIMATION_SCHEMA_VERSION, str)
    assert len(ANIMATION_SCHEMA_VERSION) > 0
