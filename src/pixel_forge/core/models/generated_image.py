"""In-memory representation returned by image generators."""

from dataclasses import dataclass

from pixel_forge.core.models.image_size import ImageSize


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """Store immutable RGB pixel data independently from any file format."""

    size: ImageSize
    pixels: bytes
    generator_name: str
    seed: int
    color_mode: str = "RGB"

    def __post_init__(self) -> None:
        expected_length = self.size.pixel_count * 3
        if self.color_mode != "RGB":
            raise ValueError("PixelForge currently supports RGB generated images only.")
        if len(self.pixels) != expected_length:
            raise ValueError(
                f"Expected {expected_length} RGB bytes, received {len(self.pixels)}."
            )
