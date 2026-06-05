"""Request model for the pixelforge animate command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.image_size import ImageSize


@dataclass(frozen=True, slots=True)
class AnimationRequest:
    """All user-supplied parameters for one animation generation operation."""

    size: ImageSize
    generator_name: str
    output_path: Path
    seed: int | None
    overwrite: bool
    options: AnimationOptions
