"""Input model for the image generation use case."""

from dataclasses import dataclass
from pathlib import Path

from pixel_forge.core.models.generation_options import GenerationOptions
from pixel_forge.core.models.image_size import ImageSize


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Describe everything required to generate and save one image."""

    size: ImageSize
    generator_name: str
    output_path: Path
    seed: int | None = None
    overwrite: bool = False
    options: GenerationOptions = GenerationOptions()
