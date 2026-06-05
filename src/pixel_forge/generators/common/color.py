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


def cosine_palette_to_rgb_float(
    position: FloatArray,
    *,
    offset: ColorTriplet = (0.5, 0.5, 0.5),
    amplitude: ColorTriplet = (0.5, 0.5, 0.5),
    frequency: ColorTriplet = (1.0, 1.0, 1.0),
    phase: ColorTriplet,
    brightness: FloatArray | None = None,
) -> FloatArray:
    """Map a scalar field to an RGB float array via the full cosine palette formula.

    color(t) = offset + amplitude * cos(2π * (frequency * t + phase))

    Returns a float64 array shaped (..., 3) in [0, 1]. Use this when you need to
    apply further post-processing (lighting, background, accent) before converting.
    """
    a = np.asarray(offset, dtype=np.float64)
    b = np.asarray(amplitude, dtype=np.float64)
    c = np.asarray(frequency, dtype=np.float64)
    d = np.asarray(phase, dtype=np.float64)
    t = position[..., np.newaxis]

    rgb: FloatArray = a + b * np.cos(math.tau * (t * c + d))

    if brightness is not None:
        rgb = rgb * np.clip(brightness, 0.0, 1.0)[..., np.newaxis]

    return rgb


def cosine_palette_to_rgb_bytes(
    position: FloatArray,
    *,
    offset: ColorTriplet = (0.5, 0.5, 0.5),
    amplitude: ColorTriplet = (0.5, 0.5, 0.5),
    frequency: ColorTriplet = (1.0, 1.0, 1.0),
    phase: ColorTriplet,
    brightness: FloatArray | None = None,
) -> UInt8Array:
    """Map a scalar field to RGB through a smooth cosine palette.

    color(t) = offset + amplitude * cos(2π * (frequency * t + phase))

    ``offset`` and ``amplitude`` correspond to the palette's cosine_a / cosine_b
    coefficients. ``frequency`` maps to cosine_c and ``phase`` to cosine_d.
    Defaults to the symmetric case (0.5 + 0.5·cos) for backward compatibility.
    """
    rgb = cosine_palette_to_rgb_float(
        position,
        offset=offset,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        brightness=brightness,
    )
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
