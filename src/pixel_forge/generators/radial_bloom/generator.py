"""Flower-like radial structures generated from polar coordinates."""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from pixel_forge.core.models import GenerationRequest
from pixel_forge.generators.common import (
    SeededArrayGenerator,
    UInt8Array,
    build_coordinate_field,
    hsv_to_rgb_bytes,
)


class RadialBloomGenerator(SeededArrayGenerator):
    """Render petal-like structures with radial and angular interference."""

    @property
    def name(self) -> str:
        return "radial-bloom"

    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        field = build_coordinate_field(request.size)
        petals = int(random_source.integers(5, 13))
        ripple_frequency = random_source.uniform(10.0, 24.0)
        phase = random_source.uniform(0.0, math.tau)
        glow_spread = random_source.uniform(0.32, 0.58)
        center_glow = np.exp(-(field.radius**2) / glow_spread)

        petal_wave = np.sin(field.angle * petals + field.radius * ripple_frequency + phase)
        ribbon_wave = np.cos(
            field.angle * (petals / 2.0) - field.radius * ripple_frequency * 0.82 - phase
        )
        fine_ripples = np.sin(field.radius * ripple_frequency * 2.4 + phase) * 0.5 + 0.5

        hue = 0.78 + 0.20 * petal_wave + 0.06 * field.radius
        saturation = 0.42 + 0.52 * center_glow + 0.06 * fine_ripples
        value = np.clip(
            0.12
            + 0.70 * center_glow
            + 0.16 * ((petal_wave + ribbon_wave + 2.0) / 4.0)
            + 0.12 * fine_ripples,
            0.0,
            1.0,
        )
        return hsv_to_rgb_bytes(hue, saturation, value)
