"""Result returned by AnimationService after a successful GIF generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pixel_forge.core.models.image_size import ImageSize


@dataclass(frozen=True, slots=True)
class AnimationResult:
    """Summary of a completed animation generation operation."""

    output_path: Path
    size: ImageSize
    generator_name: str
    master_seed: int
    animation_seed: int
    frame_count: int
    fps: int
    frame_duration_ms: int
    loop_count: int
    motion_profile: str
    bytes_written: int
    metadata_path: Path | None
    retry_index: int
    animation_rarity_tier: str
    temporal_quality_score: float
    content_id: str
