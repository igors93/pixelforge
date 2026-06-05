"""Immutable animation recipe produced before any frame rendering begins.

An AnimationRecipe extends the static ArtworkRecipe with all animated motion
decisions. Like its static counterpart, it is produced by a random-sampling
step (build_animation_recipe) and consumed by a pure rendering step
(render_frame). The renderer must never call an RNG — phase alone drives
the output.

Schema versioning follows the same convention as ArtworkRecipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pixel_forge.core.models.artwork_recipe import ArtworkRecipe

ANIMATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class AnimationRecipe:
    """All motion decisions for one animated GIF, determined before rendering.

    ``base_recipe`` is the static ArtworkRecipe shared by every frame.
    Generator-specific animation numbers live in ``generator_animation_params``.
    """

    animation_schema_version: str
    base_recipe: ArtworkRecipe
    master_seed: int
    animation_seed: int
    frame_count: int
    fps: int
    frame_duration_ms: int      # round(1000 / fps)
    loop_count: int             # 0 = forever
    motion_profile: str
    motion_intensity: float     # [0.0, 1.0]
    rotation_turns: int         # integer full turns per loop
    color_cycles: int           # integer palette cycles per loop
    pulse_count: int            # integer breathing pulses per loop
    direction: str              # "forward" | "reverse"
    animated_traits: tuple[str, ...]
    rare_animation_events: tuple[str, ...]
    retry_index: int
    generator_animation_params: Mapping[str, Any] = field(hash=False)
