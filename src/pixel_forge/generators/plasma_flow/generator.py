"""Fluid plasma generated from interacting trigonometric fields."""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from pixel_forge.core.models import GenerationRequest
from pixel_forge.generators.common import (
    SeededArrayGenerator,
    UInt8Array,
    build_coordinate_field,
    cosine_palette_to_rgb_bytes,
)


class PlasmaFlowGenerator(SeededArrayGenerator):
    """Render a smooth plasma effect with deterministic seed variation."""

    @property
    def name(self) -> str:
        return "plasma-flow"

    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        field = build_coordinate_field(request.size)
        phases = random_source.uniform(0.0, math.tau, size=4)
        center_x = random_source.uniform(-0.35, 0.35)
        center_y = random_source.uniform(-0.35, 0.35)

        shifted_radius = np.hypot(
            field.x_centered - center_x,
            field.y_centered - center_y,
        )

        plasma = (
            np.sin(field.x_unit * random_source.uniform(8.0, 14.0) * math.pi + phases[0])
            + np.sin(field.y_unit * random_source.uniform(8.0, 14.0) * math.pi + phases[1])
            + np.sin(
                (field.x_unit + field.y_unit)
                * random_source.uniform(8.0, 16.0)
                * math.pi
                + phases[2]
            )
            + np.sin(shifted_radius * random_source.uniform(14.0, 26.0) + phases[3])
        ) / 4.0

        palette_position = 0.5 + 0.5 * plasma + 0.08 * shifted_radius
        brightness = 0.46 + 0.54 * (
            0.5 + 0.5 * np.cos(plasma * math.pi + shifted_radius * 1.7)
        )
        phase_shift = random_source.uniform(0.0, 1.0)

        return cosine_palette_to_rgb_bytes(
            palette_position,
            phase=(phase_shift, phase_shift + 0.18, phase_shift + 0.52),
            frequency=(1.0, 1.1, 0.9),
            brightness=brightness,
        )
