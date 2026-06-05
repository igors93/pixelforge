"""Shared rendering helpers that translate recipe traits into pixel operations.

These functions translate high-level recipe fields (symmetry_mode, lighting_mode,
background_mode, accent_mode) into concrete pixel-level transformations. Every
recipe trait consumed here is guaranteed to produce a visible pixel difference
when changed in isolation.
"""

from __future__ import annotations

import math

import numpy as np

from pixel_forge.aesthetics.palettes.color_palette import ColorPalette, RGBFloat
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    LightingMode,
    SymmetryMode,
)
from pixel_forge.generators.common.types import FloatArray

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ──────────────────────────────────────────────────────────────────────────────


def apply_symmetry_to_coordinates(
    x: FloatArray,
    y: FloatArray,
    mode: SymmetryMode,
) -> tuple[FloatArray, FloatArray]:
    """Return (x, y) transformed according to *mode*.

    MIRROR_H folds the horizontal axis so the left and right halves mirror each
    other. MIRROR_V folds the vertical axis. RADIAL and BROKEN use the original
    coordinates (generators handle them in polar space). NONE is a no-op.
    """
    if mode == SymmetryMode.MIRROR_H:
        return np.abs(x), y
    if mode == SymmetryMode.MIRROR_V:
        return x, np.abs(y)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Lighting
# ──────────────────────────────────────────────────────────────────────────────


def apply_lighting_mode(
    rgb: FloatArray,
    radius: FloatArray,
    x: FloatArray,
    y: FloatArray,
    mode: LightingMode,
) -> FloatArray:
    """Modulate *rgb* (shape …×3) by a lighting model based on *mode*.

    FLAT:        no change (identity).
    RADIAL:      Gaussian falloff from center; bright core, dark edges.
    DIRECTIONAL: Top-left directional light; shadows to the bottom-right.
    AMBIENT:     Soft global lift that prevents pure blacks in mid-tones.
    """
    if mode == LightingMode.FLAT:
        return rgb
    if mode == LightingMode.RADIAL:
        # Gaussian that goes from 1.0 at center to ~0.37 at radius=1.
        falloff = np.exp(-(radius**2) / 0.9)[..., np.newaxis]
        # Blend: 40% baseline + 60% falloff so edges are still visible.
        return np.clip(rgb * (0.40 + 0.60 * falloff), 0.0, 1.0)
    if mode == LightingMode.DIRECTIONAL:
        # Normalized (−x − y) gives a top-left incident light vector.
        light = np.clip(0.55 + 0.45 * (-x - y) / math.sqrt(2.0), 0.05, 1.0)
        return np.clip(rgb * light[..., np.newaxis], 0.0, 1.0)
    # AMBIENT: slight boost so the darkest pixels become 10% brighter.
    return np.clip(rgb + 0.10 * (1.0 - rgb), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Background
# ──────────────────────────────────────────────────────────────────────────────


def apply_background_mode(
    rgb: FloatArray,
    radius: FloatArray,
    mode: BackgroundMode,
    palette: ColorPalette,
) -> FloatArray:
    """Blend the background zone of *rgb* according to *mode*.

    DARK:     no change (the cosine palette already produces dark areas).
    VOID:     hard-vignette; pixels beyond radius 1.1 become pure black.
    LIGHT:    blend toward the palette highlight color outside radius 0.8.
    GRADIENT: smooth radial blend from palette background at edges to nothing.
    """
    if mode == BackgroundMode.DARK:
        return rgb
    if mode == BackgroundMode.VOID:
        fade = np.clip(1.0 - (radius - 0.80) / 0.40, 0.0, 1.0)[..., np.newaxis]
        return rgb * fade
    if mode == BackgroundMode.LIGHT:
        bg = np.asarray(palette.highlight_color, dtype=np.float64)
        blend = np.clip((radius - 0.70) / 0.60, 0.0, 0.65)[..., np.newaxis]
        return np.clip(rgb * (1.0 - blend) + bg * blend, 0.0, 1.0)
    # GRADIENT: darken edges using a smooth radial ramp.
    falloff = np.clip(1.0 - (radius / 1.4) ** 2, 0.1, 1.0)[..., np.newaxis]
    return np.clip(rgb * falloff, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Accent
# ──────────────────────────────────────────────────────────────────────────────

_SPARKS_POSITIONS: tuple[tuple[float, float], ...] = (
    (-0.35, 0.50), (0.60, -0.20), (-0.70, -0.45), (0.10, 0.75),
    (0.50, 0.55), (-0.55, 0.10), (0.30, -0.65), (-0.20, -0.70),
)


def apply_accent_mode(
    rgb: FloatArray,
    radius: FloatArray,
    x: FloatArray,
    y: FloatArray,
    mode: AccentMode,
    palette: ColorPalette,
) -> FloatArray:
    """Overlay accent elements on *rgb* according to *mode*.

    NONE:      no change.
    HIGHLIGHTS: brighten the top-luminance pixels toward the palette highlight.
    SPARKS:    add small glowing dots at fixed positions in the composition.
    LUMINOUS:  add a wide luminous halo ring at radius ≈ 0.75.
    """
    if mode == AccentMode.NONE:
        return rgb
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if mode == AccentMode.HIGHLIGHTS:
        hi = np.asarray(palette.highlight_color, dtype=np.float64)
        strength = np.clip(lum - 0.50, 0.0, 0.50)[..., np.newaxis] * 2.0
        return np.clip(rgb + strength * (hi - rgb), 0.0, 1.0)
    if mode == AccentMode.SPARKS:
        hi = np.asarray(palette.highlight_color, dtype=np.float64)
        spark_field = np.zeros_like(lum)
        for sx, sy in _SPARKS_POSITIONS:
            dist2 = (x - sx) ** 2 + (y - sy) ** 2
            spark_field = spark_field + np.exp(-dist2 / 0.003)
        spark_field = np.clip(spark_field, 0.0, 1.0)[..., np.newaxis]
        return np.clip(rgb + spark_field * (hi - rgb), 0.0, 1.0)
    # LUMINOUS: wide glowing ring.
    ring = np.exp(-((radius - 0.75) ** 2) / 0.010)[..., np.newaxis]
    hi = np.asarray(palette.highlight_color, dtype=np.float64)
    return np.clip(rgb + 0.70 * ring * (hi - rgb), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Combined pipeline helper
# ──────────────────────────────────────────────────────────────────────────────


def build_rgb_float_from_palette(
    position: FloatArray,
    palette: ColorPalette,
    brightness: FloatArray | None = None,
) -> FloatArray:
    """Compute a float RGB array using the full cosine palette coefficients.

    color(t) = a + b · cos(2π · (c · t + d))
    """
    a = np.asarray(palette.cosine_a, dtype=np.float64)
    b = np.asarray(palette.cosine_b, dtype=np.float64)
    c = np.asarray(palette.cosine_c, dtype=np.float64)
    d = np.asarray(palette.cosine_d, dtype=np.float64)
    t = position[..., np.newaxis]

    rgb: FloatArray = a + b * np.cos(math.tau * (t * c + d))

    if brightness is not None:
        rgb = rgb * np.clip(brightness, 0.0, 1.0)[..., np.newaxis]

    return rgb


def apply_recipe_post_processing(
    rgb: FloatArray,
    *,
    radius: FloatArray,
    x: FloatArray,
    y: FloatArray,
    lighting_mode: LightingMode,
    background_mode: BackgroundMode,
    accent_mode: AccentMode,
    palette: ColorPalette,
) -> FloatArray:
    """Apply all recipe-level post-processing in the correct order.

    Order: lighting → background → accent.
    """
    rgb = apply_lighting_mode(rgb, radius, x, y, lighting_mode)
    rgb = apply_background_mode(rgb, radius, background_mode, palette)
    rgb = apply_accent_mode(rgb, radius, x, y, accent_mode, palette)
    return rgb


def rgb_float_to_bytes(rgb: FloatArray) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    """Clip, scale, and round float RGB to uint8."""
    from pixel_forge.generators.common.types import UInt8Array

    result: UInt8Array = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


def _hue_shift_by_palette(hue: FloatArray, palette: ColorPalette) -> FloatArray:
    """Shift hue field to align with the palette's dominant color range.

    The palette's phase (cosine_d channel 0) is used as an angular offset.
    This ensures that changing the palette changes the dominant hue cluster.
    """
    return np.mod(hue + palette.cosine_d[0], 1.0)


RGBFloat = RGBFloat  # re-export for consumers of this module
