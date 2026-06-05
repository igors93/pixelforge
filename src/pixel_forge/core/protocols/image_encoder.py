"""Contract implemented by image format encoders."""

from typing import Protocol

from pixel_forge.core.models import GeneratedImage


class ImageEncoder(Protocol):
    """Encode generated pixel data into a file-format byte stream."""

    def encode(self, image: GeneratedImage) -> bytes:
        """Return encoded image bytes."""

        ...
