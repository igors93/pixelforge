"""Reusable base classes for image generators."""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from pixel_forge.core.models import GeneratedImage, GenerationRequest
from pixel_forge.generators.common.types import UInt8Array


class SeededArrayGenerator(ABC):
    """Base class for generators that render RGB arrays with NumPy.

    Subclasses implement only their mathematical rendering logic. This class
    centralizes seed handling, validates the returned array, and converts it to
    the immutable ``GeneratedImage`` domain model.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable public name used by the CLI and registry."""

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        """Render a validated request and return immutable RGB pixel data."""

        effective_seed = request.seed if request.seed is not None else secrets.randbits(64)
        random_source = np.random.default_rng(effective_seed)
        rgb_array = np.ascontiguousarray(self.render(request, random_source))

        expected_shape = (request.size.height, request.size.width, 3)
        if rgb_array.shape != expected_shape:
            raise ValueError(
                "Generator render output must match the requested image dimensions "
                "and contain exactly three color channels."
            )
        if rgb_array.dtype != np.uint8:
            raise ValueError("Generator render output must use the uint8 dtype.")

        return GeneratedImage(
            size=request.size,
            pixels=rgb_array.tobytes(),
            generator_name=self.name,
            seed=effective_seed,
        )

    @abstractmethod
    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        """Return a ``height x width x 3`` uint8 RGB array."""
