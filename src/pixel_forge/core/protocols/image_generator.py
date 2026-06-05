"""Contract implemented by procedural image generators."""

from typing import Protocol

from pixel_forge.core.models import GeneratedImage, GenerationRequest


class ImageGenerator(Protocol):
    """Generate raw image data from a validated request."""

    @property
    def name(self) -> str:
        """Return the stable public name used by the CLI and registry."""

        ...

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        """Generate an image without performing file I/O."""

        ...
