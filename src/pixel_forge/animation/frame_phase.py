"""Deterministic frame phase generation for seamless GIF loops.

A GIF with N frames uses phases 0/N, 1/N, ..., (N-1)/N.  Phase 1.0 is
never encoded as a distinct frame because the player's natural wrap from
the last frame back to the first creates the loop boundary implicitly.
All animated transformations must be periodic with period 1 so that the
virtual frame at phase 1.0 is identical to the frame at phase 0.0.
"""

from __future__ import annotations


def generate_frame_phases(frame_count: int) -> list[float]:
    """Return a list of *frame_count* evenly-spaced phases in [0.0, 1.0).

    frame_count must be >= 2.  Phase 1.0 is never included — the GIF loop
    boundary is implicit in the player's wrap-around behaviour.

    Example for frame_count=4: [0.0, 0.25, 0.5, 0.75]
    """
    if frame_count < 2:
        raise ValueError(f"frame_count must be at least 2, got {frame_count}")
    return [i / frame_count for i in range(frame_count)]


def representative_phases() -> list[float]:
    """Four evenly-spaced phases used for palette construction and quality probes.

    Returns [0.0, 0.25, 0.5, 0.75] — a quarter-cycle sample that captures
    the full range of any 1-cycle periodic animation.
    """
    return [0.0, 0.25, 0.5, 0.75]
