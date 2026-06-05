"""User-supplied options for procedural GIF animation generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnimationOptions:
    """Options forwarded from the CLI to AnimationService.

    All values remain optional so the service can apply sensible defaults
    without coupling the CLI to domain logic.
    """

    frame_count: int = 24
    fps: int = 24
    loop_count: int = 0                     # 0 = loop forever
    motion_profile: str | None = None       # None → generator picks default
    motion_intensity: float | None = None   # None → generator picks default
    temporal_quality_threshold: float | None = None
    max_animation_retries: int = 3
    write_metadata: bool = True
    gif_colors: int = 256                   # 64 | 128 | 256
    gif_dither: str = "none"                # "none" | "floyd-steinberg"
    strict_temporal_quality: bool = False
    min_animation_rarity_tier: str | None = None
    palette_name: str | None = None         # force a specific static palette
