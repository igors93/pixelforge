"""Temporal quality evaluator for procedural GIF animations.

Evaluates a set of rendered frames for:
  - Loop seam continuity (virtual frame 0 == virtual frame N)
  - Motion energy (not static, not chaotic)
  - Temporal flicker (consistent frame transitions)
  - Per-frame static quality (worst-case check)
  - Luminance variation across frames
  - Color variation across frames

All metrics are heuristics. The aggregate score is a weighted average.
Thresholds are tunable via TemporalQualityThresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixel_forge.aesthetics.quality.quality_evaluator import QualityEvaluator, QualityThresholds
from pixel_forge.animation.quality.temporal_quality_metrics import (
    compute_color_variation,
    compute_flicker_score,
    compute_luminance_variation,
    compute_motion_energy,
    compute_seam_score,
)
from pixel_forge.core.models.temporal_quality_result import TemporalQualityResult
from pixel_forge.generators.common.types import UInt8Array


@dataclass(frozen=True, slots=True)
class TemporalQualityThresholds:
    """Configurable acceptance thresholds for temporal quality."""

    min_aggregate_score: float = 0.30
    min_seam_score: float = 0.50
    min_motion_score: float = 0.10
    min_flicker_score: float = 0.40
    min_frame_quality: float = 0.20


# Metric weights for aggregate score calculation.
_WEIGHTS: dict[str, float] = {
    "seam_score":           0.25,
    "motion_score":         0.25,
    "flicker_score":        0.20,
    "min_frame_quality":    0.15,
    "luminance_variation":  0.08,
    "color_variation":      0.07,
}


class TemporalQualityEvaluator:
    """Evaluate temporal quality of a candidate animation."""

    def __init__(
        self,
        thresholds: TemporalQualityThresholds | None = None,
    ) -> None:
        self._thresholds = thresholds or TemporalQualityThresholds()
        self._static_evaluator = QualityEvaluator(QualityThresholds())

    def evaluate(
        self,
        frames: list[UInt8Array],
        *,
        virtual_phase0: UInt8Array | None = None,
        virtual_phase1: UInt8Array | None = None,
    ) -> TemporalQualityResult:
        """Compute temporal quality metrics for the given frame sequence.

        ``virtual_phase0`` and ``virtual_phase1`` are rendered at exactly
        phase=0.0 and phase=1.0 (which should be identical for seamless loops).
        If not provided, frames[0] and frames[-1] are used for the seam check.
        """
        if len(frames) < 2:
            return self._reject("Need at least 2 frames for temporal evaluation")

        f0 = virtual_phase0 if virtual_phase0 is not None else frames[0]
        fn = virtual_phase1 if virtual_phase1 is not None else frames[-1]

        seam = compute_seam_score(f0, fn)
        motion = compute_motion_energy(frames)
        flicker = compute_flicker_score(frames)
        lum_var = compute_luminance_variation(frames)
        col_var = compute_color_variation(frames)

        # Evaluate static quality for a few representative frames.
        probe_indices = self._probe_indices(len(frames))
        frame_scores = [
            self._static_evaluator.evaluate(
                np.ascontiguousarray(frames[i])
            ).aggregate_score
            for i in probe_indices
        ]
        min_fq = float(min(frame_scores)) if frame_scores else 0.0

        metrics: dict[str, float] = {
            "seam_score":          seam,
            "motion_score":        motion,
            "flicker_score":       flicker,
            "min_frame_quality":   min_fq,
            "luminance_variation": lum_var,
            "color_variation":     col_var,
        }

        aggregate = float(sum(metrics[k] * _WEIGHTS[k] for k in _WEIGHTS))

        rejection_reasons: list[str] = []
        t = self._thresholds
        if seam < t.min_seam_score:
            rejection_reasons.append(
                f"seam_score {seam:.3f} < threshold {t.min_seam_score}"
            )
        if motion < t.min_motion_score:
            rejection_reasons.append(
                f"motion_score {motion:.3f} < threshold {t.min_motion_score}"
            )
        if flicker < t.min_flicker_score:
            rejection_reasons.append(
                f"flicker_score {flicker:.3f} < threshold {t.min_flicker_score}"
            )
        if min_fq < t.min_frame_quality:
            rejection_reasons.append(
                f"min_frame_quality {min_fq:.3f} < threshold {t.min_frame_quality}"
            )
        if aggregate < t.min_aggregate_score:
            rejection_reasons.append(
                f"aggregate {aggregate:.3f} < threshold {t.min_aggregate_score}"
            )

        accepted = len(rejection_reasons) == 0

        return TemporalQualityResult(
            aggregate_score=aggregate,
            accepted=accepted,
            seam_score=seam,
            motion_score=motion,
            flicker_score=flicker,
            min_frame_quality=min_fq,
            rejection_reasons=tuple(rejection_reasons),
            metrics=metrics,
        )

    def _reject(self, reason: str) -> TemporalQualityResult:
        return TemporalQualityResult(
            aggregate_score=0.0,
            accepted=False,
            seam_score=0.0,
            motion_score=0.0,
            flicker_score=0.0,
            min_frame_quality=0.0,
            rejection_reasons=(reason,),
            metrics={},
        )

    @staticmethod
    def _probe_indices(n: int) -> list[int]:
        """Return indices for up to 4 representative frames."""
        if n <= 4:
            return list(range(n))
        return [0, n // 4, n // 2, 3 * n // 4]
