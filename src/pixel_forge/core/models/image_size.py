"""Image dimension value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageSize:
    """Represent image dimensions without embedding validation policy."""

    width: int
    height: int

    @property
    def pixel_count(self) -> int:
        """Return the total number of pixels in the image."""

        return self.width * self.height
