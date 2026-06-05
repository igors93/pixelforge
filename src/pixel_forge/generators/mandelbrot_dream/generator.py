"""Mandelbrot fractal with smooth escape-time coloring."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from pixel_forge.core.models import GenerationRequest
from pixel_forge.generators.common import SeededArrayGenerator, UInt8Array, hsv_to_rgb_bytes


class MandelbrotDreamGenerator(SeededArrayGenerator):
    """Render a Mandelbrot set with a seed-controlled view and palette."""

    @property
    def name(self) -> str:
        return "mandelbrot-dream"

    def render(self, request: GenerationRequest, random_source: Generator) -> UInt8Array:
        width = request.size.width
        height = request.size.height
        zoom = random_source.uniform(0.85, 1.25)
        center_real = random_source.uniform(-0.78, -0.58)
        center_imag = random_source.uniform(-0.12, 0.12)
        max_iterations = 100

        aspect_ratio = width / height
        real_half_range = 1.6 / zoom * aspect_ratio
        imaginary_half_range = 1.6 / zoom

        real_axis = np.linspace(
            center_real - real_half_range,
            center_real + real_half_range,
            width,
        )
        imaginary_axis = np.linspace(
            center_imag - imaginary_half_range,
            center_imag + imaginary_half_range,
            height,
        )
        complex_plane = real_axis[None, :] + 1j * imaginary_axis[:, None]

        state = np.zeros_like(complex_plane)
        active = np.ones(complex_plane.shape, dtype=np.bool_)
        smooth_escape = np.zeros(complex_plane.shape, dtype=np.float64)

        for iteration in range(max_iterations):
            state[active] = state[active] * state[active] + complex_plane[active]
            newly_escaped = (np.abs(state) > 2.0) & active

            if np.any(newly_escaped):
                escaped_values = np.abs(state[newly_escaped])
                smooth_escape[newly_escaped] = (
                    iteration + 1 - np.log2(np.log2(escaped_values))
                )
                active[newly_escaped] = False

            if not np.any(active):
                break

        escaped_mask = ~active
        normalized = np.zeros_like(smooth_escape)
        if np.any(escaped_mask):
            normalized[escaped_mask] = (
                smooth_escape[escaped_mask] / smooth_escape[escaped_mask].max()
            )

        hue_offset = random_source.uniform(0.0, 0.25)
        hue = hue_offset + 0.85 * normalized
        saturation = np.where(active, 0.0, 0.88)
        value = np.where(active, 0.035, 0.16 + 0.84 * normalized**0.68)
        return hsv_to_rgb_bytes(hue, saturation, value)
