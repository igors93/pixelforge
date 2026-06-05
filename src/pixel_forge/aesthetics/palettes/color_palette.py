"""Named immutable palette definition.

Each palette stores both reference sRGB colors (background, shadow, highlight)
and cosine palette coefficients for smooth mathematical color mapping. The
cosine palette formula is:

    color(t) = a + b * cos(2π * (c * t + d))

where t ∈ [0, 1] and a, b, c, d are per-channel RGB vectors. This formula
produces continuous, cyclic gradients used by the cosine_palette_to_rgb_bytes
helper already present in the codebase.

All RGB tuples use the [0, 1] range. Final pixel output clips and converts to
uint8 in the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

RGBFloat = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ColorPalette:
    """A named, immutable palette with perceptually motivated color parameters."""

    name: str
    # Cosine palette coefficients (per RGB channel)
    cosine_a: RGBFloat   # offset (baseline brightness per channel)
    cosine_b: RGBFloat   # amplitude
    cosine_c: RGBFloat   # frequency
    cosine_d: RGBFloat   # phase
    # Reference colors for lighting and background
    background_color: RGBFloat
    shadow_color: RGBFloat
    highlight_color: RGBFloat
    # Perceptual limits
    min_saturation: float       # [0, 1]
    max_saturation: float       # [0, 1]
    min_luminance: float        # [0, 1]
    max_luminance: float        # [0, 1]
    # Sampling and rarity metadata
    sampling_probability: float # relative weight in palette sampling
    compatible_generators: tuple[str, ...]  # empty tuple means compatible with all
    description: str = ""
