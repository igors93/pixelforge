"""Implementation of the first PixelForge generation algorithm."""

import random
import secrets

from pixel_forge.core.models import GeneratedImage, GenerationRequest


class RandomNoiseGenerator:
    """Generate independent uniformly distributed RGB values for every pixel.

    A local pseudo-random number generator avoids mutating Python's global
    random state. When a seed is provided, the complete output is reproducible.
    """

    @property
    def name(self) -> str:
        return "random-noise"

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        # An automatically generated seed is returned with the result so that
        # every image can be reproduced later.
        effective_seed = request.seed if request.seed is not None else secrets.randbits(64)
        random_source = random.Random(effective_seed)

        byte_count = request.size.pixel_count * 3
        pixels = random_source.randbytes(byte_count)

        return GeneratedImage(
            size=request.size,
            pixels=pixels,
            generator_name=self.name,
            seed=effective_seed,
        )
