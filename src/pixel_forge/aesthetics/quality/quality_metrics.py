"""Vectorized quality metric calculations for RGB image arrays.

All functions operate on NumPy arrays and avoid Python-level pixel loops.
The functions return normalized scalar values in [0, 1] unless documented
otherwise. NaN and infinity are protected against at every calculation site.
"""

from __future__ import annotations

import math

import numpy as np

from pixel_forge.generators.common.types import FloatArray, UInt8Array


def luminance_from_rgb(rgb: UInt8Array) -> FloatArray:
    """Compute perceptual luminance (BT.709) for each pixel in [0, 1]."""
    f = rgb.astype(np.float64) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def compute_luminance_contrast(lum: FloatArray) -> float:
    """Return the normalized standard deviation of luminance (contrast proxy)."""
    return float(np.clip(np.std(lum) * 4.0, 0.0, 1.0))


def compute_clipped_black_ratio(lum: FloatArray, threshold: float = 0.04) -> float:
    """Fraction of pixels with luminance at or below *threshold*."""
    return float(np.mean(lum <= threshold))


def compute_clipped_white_ratio(lum: FloatArray, threshold: float = 0.96) -> float:
    """Fraction of pixels with luminance at or above *threshold*."""
    return float(np.mean(lum >= threshold))


def compute_saturation_metrics(rgb: UInt8Array) -> tuple[float, float]:
    """Return (mean_saturation, saturation_spread) in [0, 1].

    Saturation is approximated from the HSI model without per-pixel Python loops:
        S = 1 - 3 * min(R, G, B) / (R + G + B + ε)
    """
    f = rgb.astype(np.float64) / 255.0
    channel_min = f.min(axis=-1)
    channel_sum = f.sum(axis=-1)
    saturation = np.where(
        channel_sum > 1e-6,
        1.0 - 3.0 * channel_min / (channel_sum + 1e-6),
        0.0,
    )
    saturation = np.clip(saturation, 0.0, 1.0)
    mean_sat = float(np.mean(saturation))
    spread_sat = float(np.clip(np.std(saturation) * 2.0, 0.0, 1.0))
    return mean_sat, spread_sat


def compute_color_diversity(rgb: UInt8Array, buckets: int = 32) -> float:
    """Fraction of hue angle buckets that are occupied across the image.

    Pixels below a saturation threshold are excluded (grey pixels have no
    meaningful hue). Higher values indicate a richer color range.
    """
    f = rgb.astype(np.float64) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    delta = mx - mn

    hue = np.zeros_like(mx)
    mask_r = (mx == r) & (delta > 0)
    mask_g = (mx == g) & (delta > 0)
    mask_b = (mx == b) & (delta > 0)

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-9)) % 6.0
    hue[mask_g] = (b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-9) + 2.0
    hue[mask_b] = (r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-9) + 4.0

    saturation = np.where(mx > 1e-6, delta / (mx + 1e-6), 0.0)
    saturated_mask = saturation > 0.10

    if not np.any(saturated_mask):
        return 0.0

    hue_degrees = (hue[saturated_mask] / 6.0 * buckets).astype(np.int32) % buckets
    occupied = len(np.unique(hue_degrees))
    return occupied / buckets


def compute_visual_entropy(lum: FloatArray, bins: int = 64) -> float:
    """Normalized Shannon entropy of the luminance histogram.

    Returns 0 when all pixels have the same luminance, and 1 when the
    luminance is uniformly distributed across *bins* buckets.
    """
    counts, _ = np.histogram(lum, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log2(p)))
    max_entropy = math.log2(bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_edge_density(lum: FloatArray) -> float:
    """Fraction of interior pixels classified as edges via a Sobel approximation.

    The Sobel gradient is computed from adjacent rows/columns without external
    dependencies. Returns values in [0, 1]; very high values indicate noise.
    """
    if lum.shape[0] < 3 or lum.shape[1] < 3:
        return 0.0

    gx = lum[:, 2:] - lum[:, :-2]
    gy = lum[2:, :] - lum[:-2, :]

    rows = min(gx.shape[0], gy.shape[0])
    cols = min(gx.shape[1], gy.shape[1])
    magnitude = np.hypot(gx[:rows, :cols], gy[:rows, :cols])

    edge_threshold = 0.08
    return float(np.mean(magnitude > edge_threshold))


def compute_center_border_balance(lum: FloatArray) -> float:
    """Ratio of border energy to total energy; 0.5 is ideal balance."""
    h, w = lum.shape
    if h < 4 or w < 4:
        return 0.5
    border_width = max(1, min(h, w) // 8)

    border_mask = np.zeros_like(lum, dtype=np.bool_)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    total_energy = float(np.sum(lum))
    if total_energy < 1e-9:
        return 0.5
    border_energy = float(np.sum(lum[border_mask]))
    return float(np.clip(border_energy / total_energy, 0.0, 1.0))


def compute_symmetry(lum: FloatArray) -> tuple[float, float]:
    """Return (horizontal_symmetry, vertical_symmetry) in [0, 1].

    Symmetry is the normalized cross-correlation between the two halves.
    A value of 1 means the two halves are identical; 0 means unrelated.
    """

    def _half_correlation(a: FloatArray, b: FloatArray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        if a_flat.size == 0 or b_flat.size == 0:
            return 0.0
        # Pad or truncate to the same length
        n = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:n]
        b_flat = b_flat[:n]
        std_a = float(np.std(a_flat))
        std_b = float(np.std(b_flat))
        if std_a < 1e-9 or std_b < 1e-9:
            return 0.0
        corr = float(np.mean((a_flat - np.mean(a_flat)) * (b_flat - np.mean(b_flat))))
        return float(np.clip((corr / (std_a * std_b) + 1.0) / 2.0, 0.0, 1.0))

    h, w = lum.shape
    h_sym = _half_correlation(lum[:, : w // 2], np.fliplr(lum)[:, : w // 2])
    v_sym = _half_correlation(lum[: h // 2, :], np.flipud(lum)[: h // 2, :])
    return h_sym, v_sym
