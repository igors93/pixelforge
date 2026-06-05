"""Individual temporal quality metric calculations.

Each function accepts a sequence of RGB uint8 frames and returns a normalised
float in [0.0, 1.0] where 1.0 is the best possible value. Functions are pure
— no side effects, no RNG, no I/O.

Limitations are stated inline so callers set appropriate expectations.
These are heuristics, not objective measures of aesthetic quality.
"""

from __future__ import annotations

import math

import numpy as np

from pixel_forge.generators.common.types import UInt8Array


def compute_seam_score(frame0: UInt8Array, last_frame: UInt8Array) -> float:
    """Measure loop seam quality by comparing the first and last encoded frames.

    A seamless loop has identical or near-identical boundary frames (high score).
    A harsh jump has a large RMSE (low score).

    Limitation: measures pixel-level discontinuity only, not velocity continuity.
    """
    diff = frame0.astype(np.float64) - last_frame.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    # RMSE of 0 → score 1.0; RMSE of 50+ (out of 255) → score ~0.0.
    score = math.exp(-rmse / 25.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_motion_energy(frames: list[UInt8Array]) -> float:
    """Estimate motion energy as mean frame-to-frame RMSE, normalised to [0, 1].

    Near-static animations (energy ≈ 0) score low; very chaotic animations
    (energy > 40 on 0-255 scale) also score lower via a soft cap.
    Ideal range ≈ 5–30 RMSE units.
    """
    if len(frames) < 2:
        return 0.0

    rmse_values: list[float] = []
    for i in range(len(frames) - 1):
        diff = frames[i].astype(np.float64) - frames[i + 1].astype(np.float64)
        rmse_values.append(float(np.sqrt(np.mean(diff ** 2))))

    mean_rmse = float(np.mean(rmse_values))

    # Penalise near-static (energy < 2) and excessively chaotic (energy > 50).
    if mean_rmse < 1.0:
        return 0.0
    if mean_rmse > 50.0:
        return float(np.clip(1.0 - (mean_rmse - 50.0) / 100.0, 0.0, 0.5))
    # Smooth ramp: 0 at rmse=0, peaks near rmse≈15, tapers above 30.
    score = math.exp(-((mean_rmse - 15.0) ** 2) / (2 * 20.0 ** 2))
    return float(np.clip(score + 0.15, 0.0, 1.0))


def compute_flicker_score(frames: list[UInt8Array]) -> float:
    """Measure temporal consistency by checking variance of frame-to-frame RMSE.

    High variance in frame transitions (some frames very different, others
    identical) indicates flickering. Low variance = smooth motion.

    Returns 1.0 for perfectly consistent motion, 0.0 for extreme flickering.
    """
    if len(frames) < 3:
        return 1.0

    rmse_values: list[float] = []
    for i in range(len(frames) - 1):
        diff = frames[i].astype(np.float64) - frames[i + 1].astype(np.float64)
        rmse_values.append(float(np.sqrt(np.mean(diff ** 2))))

    variance = float(np.var(rmse_values))
    # Low variance → score near 1.0; high variance → score near 0.0.
    score = math.exp(-variance / 50.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_luminance_variation(frames: list[UInt8Array]) -> float:
    """Check that mean luminance varies meaningfully across frames.

    An animation where every frame has the same mean brightness scores low.
    Returns a score in [0, 1] where 1 = good variation.
    """
    if len(frames) < 2:
        return 0.0

    mean_lums: list[float] = []
    for f in frames:
        lum = 0.2126 * f[..., 0].astype(np.float64) + \
              0.7152 * f[..., 1].astype(np.float64) + \
              0.0722 * f[..., 2].astype(np.float64)
        mean_lums.append(float(np.mean(lum)))

    lum_range = max(mean_lums) - min(mean_lums)
    # Range of ≥10 luminance units (out of 255) scores well.
    return float(np.clip(lum_range / 20.0, 0.0, 1.0))


def compute_color_variation(frames: list[UInt8Array]) -> float:
    """Check that mean hue / saturation varies across frames.

    Computes the standard deviation of per-frame mean R, G, B channels and
    reports how spread they are. Scores 1.0 if channels vary by ≥8 units.
    """
    if len(frames) < 2:
        return 0.0

    channel_means: list[tuple[float, float, float]] = []
    for f in frames:
        channel_means.append((
            float(np.mean(f[..., 0])),
            float(np.mean(f[..., 1])),
            float(np.mean(f[..., 2])),
        ))

    r_std = float(np.std([c[0] for c in channel_means]))
    g_std = float(np.std([c[1] for c in channel_means]))
    b_std = float(np.std([c[2] for c in channel_means]))

    mean_std = (r_std + g_std + b_std) / 3.0
    return float(np.clip(mean_std / 8.0, 0.0, 1.0))
