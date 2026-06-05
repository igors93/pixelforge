"""Reusable periodic math helpers for mathematically seamless GIF loops.

All functions operate on a normalized phase p ∈ [0.0, 1.0).  Animating N
frames at phase_i = i / N guarantees that the GIF player's jump from frame
N-1 back to frame 0 is continuous — no duplicate endpoint, no seam.

Design rules enforced here:
  * Every path must be closed: f(0) == f(1) exactly.
  * Integer cycle/turn counts ensure this without floating-point hacks.
  * No ease-in/ease-out unless the curve returns exactly to its start value.
"""

from __future__ import annotations

import math


def loop_angle(phase: float) -> float:
    """Full circle angle for the given phase: τ × phase ∈ [0, τ)."""
    return math.tau * phase


def cyclic_sine(phase: float, *, cycles: int = 1) -> float:
    """Sine evaluated at τ × cycles × phase.

    With integer cycles the path closes at phase=1.0: sin(τ × n × 1) == 0.
    """
    return math.sin(math.tau * cycles * phase)


def cyclic_cosine(phase: float, *, cycles: int = 1) -> float:
    """Cosine evaluated at τ × cycles × phase.

    With integer cycles the path closes at phase=1.0: cos(τ × n × 1) == 1.
    """
    return math.cos(math.tau * cycles * phase)


def periodic_rotation(phase: float, *, turns: int = 1) -> float:
    """Rotation angle in radians for the given phase: τ × turns × phase.

    With integer turns the rotation returns to its starting angle at phase=1.
    """
    return math.tau * turns * phase


def periodic_color_shift(phase: float, *, cycles: int = 1) -> float:
    """Normalized [0, 1) color phase shift: frac(cycles × phase).

    Wrapping via modulo guarantees the palette cycles smoothly.
    """
    return math.fmod(cycles * phase, 1.0)


def circular_orbit(
    phase: float,
    *,
    cx: float,
    cy: float,
    radius: float,
) -> tuple[float, float]:
    """(x, y) position on a circle of *radius* centred at (cx, cy).

    Completes exactly one full orbit per unit of phase.
    """
    angle = loop_angle(phase)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def periodic_pulse(phase: float, *, pulses: int = 1) -> float:
    """Smooth [0, 1] pulse that peaks once per pulse cycle.

    Returns (1 + cos(τ × pulses × phase)) / 2, which is 1 at peak
    and 0 at trough. With integer pulses the path is closed.
    """
    return (1.0 + math.cos(math.tau * pulses * phase)) / 2.0


def smooth_periodic_envelope(
    phase: float,
    *,
    cycles: int = 1,
    bias: float = 0.0,
    amplitude: float = 1.0,
) -> float:
    """Smooth oscillation around *bias* with given *amplitude*.

    Returns bias + amplitude × sin(τ × cycles × phase).
    Closed for any integer cycles.
    """
    return bias + amplitude * math.sin(math.tau * cycles * phase)
