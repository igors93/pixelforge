"""Result model for heuristic image quality evaluation.

The quality evaluator measures transparent, mathematical properties of the
rendered RGB array (contrast, saturation spread, edge density, etc.). It does
not claim to measure beauty. Its purpose is to identify clearly weak outputs –
for example, completely dark images or zero-saturation renders – so that the
retry system can produce a better candidate before saving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QualityResult:
    """Heuristic quality metrics computed from the rendered RGB array."""

    luminance_contrast: float    # [0, 1] – 0 = flat, 1 = maximum spread
    clipped_black_ratio: float   # fraction of pixels at or below threshold
    clipped_white_ratio: float   # fraction of pixels at or above threshold
    mean_saturation: float       # average perceived saturation [0, 1]
    saturation_spread: float     # std-dev of saturation across image [0, 1]
    color_diversity: float       # fraction of distinct hue buckets occupied [0, 1]
    visual_entropy: float        # normalized Shannon entropy of luminance histogram
    edge_density: float          # fraction of pixels classified as edges [0, 1]
    center_border_balance: float # 0 = all energy at center, 1 = all at border, 0.5 = balanced
    horizontal_symmetry: float   # [0, 1] correlation between left and right halves
    vertical_symmetry: float     # [0, 1] correlation between top and bottom halves
    aggregate_score: float       # combined score in [0, 1]
    accepted: bool               # True if the image passes the configured thresholds
    rejection_reasons: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(hash=False, default_factory=dict)
