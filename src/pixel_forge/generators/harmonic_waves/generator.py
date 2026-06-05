"""Layered harmonic fields with domain warping and cosine coloring."""

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


class HarmonicWavesGenerator(SeededArrayGenerator):
    """Render flowing structures from interacting trigonometric fields.

    The algorithm rotates the coordinate plane, applies a smooth domain warp,
    combines linear, radial, and angular waves, and finally maps the resulting
    scalar field through a cosine color palette. A seed changes the composition
    without introducing unstructured per-pixel noise.
    """

    @property
    def name(self) -> str:
        return "harmonic-waves"

    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        field = build_coordinate_field(request.size)

        rotation = random_source.uniform(-math.pi, math.pi)
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        rotated_x = field.x_centered * cosine - field.y_centered * sine
        rotated_y = field.x_centered * sine + field.y_centered * cosine

        phases = random_source.uniform(0.0, math.tau, size=4)
        warp_strength = random_source.uniform(0.18, 0.38)
        warp_frequency_x = random_source.uniform(2.5, 5.0)
        warp_frequency_y = random_source.uniform(2.5, 5.0)

        # Domain warping bends otherwise regular sine waves into organic curves.
        warped_x = rotated_x + warp_strength * np.sin(
            rotated_y * warp_frequency_y + phases[0]
        )
        warped_y = rotated_y + warp_strength * np.cos(
            rotated_x * warp_frequency_x + phases[1]
        )

        radius = np.hypot(warped_x, warped_y)
        angle = np.arctan2(warped_y, warped_x)

        wave_x = np.sin(
            warped_x * random_source.uniform(5.0, 9.0)
            + 1.6 * np.sin(warped_y * random_source.uniform(2.0, 4.0) + phases[1])
            + phases[0]
        )
        wave_y = np.cos(
            warped_y * random_source.uniform(5.0, 9.0)
            + 1.6 * np.cos(warped_x * random_source.uniform(2.0, 4.0) + phases[2])
            + phases[1]
        )
        spiral = np.sin(
            radius * random_source.uniform(10.0, 18.0)
            + angle * int(random_source.integers(3, 9))
            + phases[2]
        )
        lattice_frequency = random_source.uniform(3.0, 6.5)
        lattice = np.sin((warped_x + warped_y) * lattice_frequency + phases[3]) * np.cos(
            (warped_x - warped_y) * lattice_frequency * 0.82 - phases[0]
        )

        combined = 0.34 * wave_x + 0.30 * wave_y + 0.24 * spiral + 0.12 * lattice
        palette_position = 0.5 + 0.5 * np.sin(combined * 2.25 + radius * 0.65)

        # A secondary field controls light independently from hue and gives the
        # pattern depth without relying on random brightness noise.
        brightness = 0.52 + 0.48 * (
            0.5 + 0.5 * np.cos(combined * math.pi - spiral * 0.75)
        )

        palette_templates = (
            (0.00, 0.14, 0.32),
            (0.00, 0.33, 0.67),
            (0.05, 0.22, 0.55),
            (0.08, 0.48, 0.78),
        )
        palette_index = int(random_source.integers(0, len(palette_templates)))
        phase_shift = random_source.uniform(0.0, 1.0)
        palette_template = palette_templates[palette_index]
        palette_phase = (
            palette_template[0] + phase_shift,
            palette_template[1] + phase_shift,
            palette_template[2] + phase_shift,
        )

        return cosine_palette_to_rgb_bytes(
            palette_position,
            phase=palette_phase,
            frequency=(1.0, 1.0, 1.0),
            brightness=brightness,
        )
