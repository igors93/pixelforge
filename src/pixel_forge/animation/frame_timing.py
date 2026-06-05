"""Timing helpers for GIF animation frames."""

from __future__ import annotations

GIF_TIME_UNIT_MS = 10


def quantize_gif_duration_ms(duration_ms: int) -> int:
    """Convert a duration to the nearest value representable by GIF.

    GIF stores frame durations in hundredths of a second, so durations
    must be multiples of 10 milliseconds.
    """
    if duration_ms <= 0:
        raise ValueError(
            f"duration_ms must be greater than zero, got {duration_ms}"
        )

    units = round(duration_ms / GIF_TIME_UNIT_MS)

    return max(GIF_TIME_UNIT_MS, units * GIF_TIME_UNIT_MS)


def frame_duration_ms_from_fps(fps: int) -> int:
    """Return a GIF-compatible frame duration for the requested FPS."""
    if fps <= 0:
        raise ValueError(f"fps must be greater than zero, got {fps}")

    requested_duration = round(1000 / fps)

    return quantize_gif_duration_ms(requested_duration)