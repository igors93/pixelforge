"""Vectorized color helpers for procedural generators."""

from __future__ import annotations

import math

import numpy as np

from pixel_forge.generators.common.types import FloatArray, UInt8Array

ColorTriplet = tuple[float, float, float]


def hsv_to_rgb_bytes(
    hue: FloatArray,
    saturation: FloatArray,
    value: FloatArray,
) -> UInt8Array:
    """Convert HSV fields in the 0..1 range to a uint8 RGB array."""

    hue = np.mod(hue, 1.0)
    saturation = np.clip(saturation, 0.0, 1.0)
    value = np.clip(value, 0.0, 1.0)

    sector = np.floor(hue * 6.0).astype(np.int16)
    fraction = hue * 6.0 - sector
    sector = np.mod(sector, 6)

    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)

    red = np.empty_like(value)
    green = np.empty_like(value)
    blue = np.empty_like(value)

    mask = sector == 0
    red[mask], green[mask], blue[mask] = value[mask], t[mask], p[mask]

    mask = sector == 1
    red[mask], green[mask], blue[mask] = q[mask], value[mask], p[mask]

    mask = sector == 2
    red[mask], green[mask], blue[mask] = p[mask], value[mask], t[mask]

    mask = sector == 3
    red[mask], green[mask], blue[mask] = p[mask], q[mask], value[mask]

    mask = sector == 4
    red[mask], green[mask], blue[mask] = t[mask], p[mask], value[mask]

    mask = sector == 5
    red[mask], green[mask], blue[mask] = value[mask], p[mask], q[mask]

    rgb = np.stack((red, green, blue), axis=-1)
    return np.rint(rgb * 255.0).astype(np.uint8)


def cosine_palette_to_rgb_bytes(
    position: FloatArray,
    *,
    phase: ColorTriplet,
    frequency: ColorTriplet = (1.0, 1.0, 1.0),
    brightness: FloatArray | None = None,
) -> UInt8Array:
    """Map a scalar field to RGB through a smooth cosine palette.

    A cosine palette keeps neighboring values visually continuous while still
    allowing saturated, contrasting colors. ``phase`` controls the palette and
    ``frequency`` controls how quickly each color channel cycles.
    """

    phase_vector = np.asarray(phase, dtype=np.float64)
    frequency_vector = np.asarray(frequency, dtype=np.float64)
    palette_position = position[..., np.newaxis]

    rgb = 0.5 + 0.5 * np.cos(
        math.tau * (palette_position * frequency_vector + phase_vector)
    )

    if brightness is not None:
        rgb *= np.clip(brightness, 0.0, 1.0)[..., np.newaxis]

    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
