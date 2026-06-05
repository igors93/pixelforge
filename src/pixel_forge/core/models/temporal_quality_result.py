"""Result of temporal quality evaluation for an animation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalQualityResult:
    """Temporal quality metrics for one candidate animation."""

    aggregate_score: float
    accepted: bool
    seam_score: float           # loop seam: 1.0 = perfect, 0.0 = total mismatch
    motion_score: float         # energy of frame-to-frame change; penalises near-static
    flicker_score: float        # 1.0 = smooth, 0.0 = excessive variance
    min_frame_quality: float    # worst single-frame static quality score
    rejection_reasons: tuple[str, ...]
    metrics: Mapping[str, float]
