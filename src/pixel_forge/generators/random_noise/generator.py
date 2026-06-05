"""Implementation of the baseline random-noise generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from pixel_forge.core.models import GenerationRequest
from pixel_forge.generators.common import SeededArrayGenerator, UInt8Array


class RandomNoiseGenerator(SeededArrayGenerator):
    """Generate independent uniformly distributed RGB values for every pixel.

    The generator remains useful as a baseline, test fixture, and debugging
    tool after the addition of structured mathematical generators.
    """

    @property
    def name(self) -> str:
        return "random-noise"

    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        return random_source.integers(
            0,
            256,
            size=(request.size.height, request.size.width, 3),
            dtype=np.uint8,
        )
