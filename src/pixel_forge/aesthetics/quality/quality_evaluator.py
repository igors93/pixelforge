"""Heuristic quality evaluator for rendered RGB image arrays.

This evaluator measures transparent, mathematical properties of the image. It
does not claim to measure beauty. Its purpose is to detect clearly weak outputs
(all-black images, zero-saturation renders, pure noise) so the retry system can
request a better candidate before writing to disk.

All thresholds are configurable via QualityThresholds. The aggregate score is
a weighted average of the individual metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixel_forge.aesthetics.quality.quality_metrics import (
    compute_center_border_balance,
    compute_clipped_black_ratio,
    compute_clipped_white_ratio,
    compute_color_diversity,
    compute_edge_density,
    compute_luminance_contrast,
    compute_saturation_metrics,
    compute_symmetry,
    compute_visual_entropy,
    luminance_from_rgb,
)
from pixel_forge.core.models.quality_result import QualityResult
from pixel_forge.generators.common.types import UInt8Array


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Configurable acceptance thresholds for quality metrics."""

    min_luminance_contrast: float = 0.05   # below this → flat image
    max_clipped_black_ratio: float = 0.70  # too dark
    max_clipped_white_ratio: float = 0.70  # too bright
    min_mean_saturation: float = 0.02      # monochrome is ok but not zero
    min_color_diversity: float = 0.0       # 0 = any diversity ok (monochrome palettes exist)
    min_visual_entropy: float = 0.10       # very low entropy = stuck single luminance
    min_aggregate_score: float = 0.30      # overall rejection threshold


_METRIC_WEIGHTS: dict[str, float] = {
    "luminance_contrast": 0.20,
    "mean_saturation": 0.15,
    "saturation_spread": 0.10,
    "color_diversity": 0.10,
    "visual_entropy": 0.20,
    "edge_density_normalized": 0.10,
    "center_border_balance_score": 0.15,
}


class QualityEvaluator:
    """Evaluate the visual quality of a rendered RGB array."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = thresholds if thresholds is not None else QualityThresholds()

    def evaluate(self, rgb: UInt8Array) -> QualityResult:
        """Compute quality metrics and determine acceptance."""
        lum = luminance_from_rgb(rgb)

        contrast = compute_luminance_contrast(lum)
        black_ratio = compute_clipped_black_ratio(lum)
        white_ratio = compute_clipped_white_ratio(lum)
        mean_sat, sat_spread = compute_saturation_metrics(rgb)
        diversity = compute_color_diversity(rgb)
        entropy = compute_visual_entropy(lum)
        edge_density = compute_edge_density(lum)
        cbb = compute_center_border_balance(lum)
        h_sym, v_sym = compute_symmetry(lum)

        t = self._thresholds
        rejection_reasons: list[str] = []

        if contrast < t.min_luminance_contrast:
            rejection_reasons.append(
                f"Luminance contrast {contrast:.3f} below minimum {t.min_luminance_contrast}."
            )
        if black_ratio > t.max_clipped_black_ratio:
            rejection_reasons.append(
                f"Clipped-black ratio {black_ratio:.3f} exceeds {t.max_clipped_black_ratio}."
            )
        if white_ratio > t.max_clipped_white_ratio:
            rejection_reasons.append(
                f"Clipped-white ratio {white_ratio:.3f} exceeds {t.max_clipped_white_ratio}."
            )
        if mean_sat < t.min_mean_saturation:
            rejection_reasons.append(
                f"Mean saturation {mean_sat:.3f} below minimum {t.min_mean_saturation}."
            )
        if entropy < t.min_visual_entropy:
            rejection_reasons.append(
                f"Visual entropy {entropy:.3f} below minimum {t.min_visual_entropy}."
            )

        # Normalize edge density: 0.1–0.5 is healthy; outside that range scores lower.
        edge_norm = float(np.clip(1.0 - abs(edge_density - 0.25) * 4.0, 0.0, 1.0))

        # Balance score: distance from 0.5 (ideal) normalized to [0, 1].
        balance_score = float(1.0 - abs(cbb - 0.5) * 2.0)

        component_scores: dict[str, float] = {
            "luminance_contrast": contrast,
            "mean_saturation": mean_sat,
            "saturation_spread": sat_spread,
            "color_diversity": diversity,
            "visual_entropy": entropy,
            "edge_density_normalized": edge_norm,
            "center_border_balance_score": balance_score,
        }

        aggregate = sum(
            _METRIC_WEIGHTS[k] * component_scores[k] for k in _METRIC_WEIGHTS
        )
        aggregate = float(np.clip(aggregate, 0.0, 1.0))

        if aggregate < t.min_aggregate_score:
            rejection_reasons.append(
                f"Aggregate score {aggregate:.3f} below minimum {t.min_aggregate_score}."
            )

        all_metrics: dict[str, float] = dict(component_scores)
        all_metrics.update({
            "clipped_black_ratio": black_ratio,
            "clipped_white_ratio": white_ratio,
            "edge_density": edge_density,
            "center_border_balance": cbb,
            "horizontal_symmetry": h_sym,
            "vertical_symmetry": v_sym,
        })

        return QualityResult(
            luminance_contrast=contrast,
            clipped_black_ratio=black_ratio,
            clipped_white_ratio=white_ratio,
            mean_saturation=mean_sat,
            saturation_spread=sat_spread,
            color_diversity=diversity,
            visual_entropy=entropy,
            edge_density=edge_density,
            center_border_balance=cbb,
            horizontal_symmetry=h_sym,
            vertical_symmetry=v_sym,
            aggregate_score=aggregate,
            accepted=not rejection_reasons,
            rejection_reasons=tuple(rejection_reasons),
            metrics=all_metrics,
        )
