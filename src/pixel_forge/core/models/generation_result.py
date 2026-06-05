"""Output model for a completed generation operation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pixel_forge.core.models.image_size import ImageSize


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Summarize the generated artifact for CLI and API consumers."""

    output_path: Path
    size: ImageSize
    generator_name: str
    seed: int
    bytes_written: int
    metadata_path: Path | None = None   # JSON manifest path, or None if disabled
    retry_index: int = 0
    overall_rarity_tier: str = "Common"
    quality_score: float = 0.0
    content_id: str = ""                # SHA-256 derived content identifier
