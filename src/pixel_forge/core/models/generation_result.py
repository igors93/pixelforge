"""Output model for a completed generation operation."""

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
