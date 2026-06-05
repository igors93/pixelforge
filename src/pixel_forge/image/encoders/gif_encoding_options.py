"""Typed options for the GIF encoder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GifEncodingOptions:
    """Options consumed by GifEncoder.

    ``gif_colors`` must be a power of 2 between 2 and 256.
    ``dither`` is "none" or "floyd-steinberg".
    ``loop_count`` is 0 for infinite looping or a positive integer.
    """

    gif_colors: int = 256
    dither: str = "none"         # "none" | "floyd-steinberg"
    loop_count: int = 0          # 0 = infinite
