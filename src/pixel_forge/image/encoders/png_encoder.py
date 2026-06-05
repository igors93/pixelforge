"""PNG encoder backed by Pillow."""

from io import BytesIO

from PIL import Image

from pixel_forge.core.models import GeneratedImage


class PngEncoder:
    """Convert an RGB generated image into valid PNG bytes."""

    def __init__(self, *, compress_level: int = 6) -> None:
        if not 0 <= compress_level <= 9:
            raise ValueError("PNG compression level must be between 0 and 9.")
        self._compress_level = compress_level

    def encode(self, image: GeneratedImage) -> bytes:
        output = BytesIO()

        # frombytes performs one compact conversion from our domain model into
        # Pillow without a Python-level loop over individual pixels.
        pillow_image = Image.frombytes(
            image.color_mode,
            (image.size.width, image.size.height),
            image.pixels,
        )
        try:
            pillow_image.save(
                output,
                format="PNG",
                compress_level=self._compress_level,
                optimize=False,
            )
            return output.getvalue()
        finally:
            pillow_image.close()
            output.close()
