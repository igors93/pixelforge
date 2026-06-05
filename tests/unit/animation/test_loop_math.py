"""Unit tests for loop_math periodic helpers.

Every helper must be closed: f(0.0) == f(1.0) for integer cycle counts.
"""

from __future__ import annotations

import math

import pytest

from pixel_forge.animation.loop_math import (
    circular_orbit,
    cyclic_cosine,
    cyclic_sine,
    loop_angle,
    periodic_color_shift,
    periodic_pulse,
    periodic_rotation,
    smooth_periodic_envelope,
)


def test_loop_angle_zero() -> None:
    assert loop_angle(0.0) == 0.0


def test_loop_angle_one_full_circle() -> None:
    assert math.isclose(loop_angle(1.0), math.tau, rel_tol=1e-12)


def test_loop_angle_half() -> None:
    assert math.isclose(loop_angle(0.5), math.pi, rel_tol=1e-12)


@pytest.mark.parametrize("cycles", [1, 2, 3, 5])
def test_cyclic_sine_closed(cycles: int) -> None:
    """sin path closes: sin(0) == sin(τ×cycles×1.0) == sin(τ×n) == 0."""
    assert math.isclose(cyclic_sine(0.0, cycles=cycles), 0.0, abs_tol=1e-12)
    assert math.isclose(cyclic_sine(1.0, cycles=cycles), 0.0, abs_tol=1e-12)


def test_cyclic_sine_quarter_period() -> None:
    assert math.isclose(cyclic_sine(0.25, cycles=1), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("cycles", [1, 2, 3])
def test_cyclic_cosine_closed(cycles: int) -> None:
    """cos path closes: cos(0) == cos(τ×cycles×1.0) == 1."""
    assert math.isclose(cyclic_cosine(0.0, cycles=cycles), 1.0, abs_tol=1e-12)
    assert math.isclose(cyclic_cosine(1.0, cycles=cycles), 1.0, abs_tol=1e-12)


def test_cyclic_cosine_half_period() -> None:
    assert math.isclose(cyclic_cosine(0.5, cycles=1), -1.0, rel_tol=1e-9)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_periodic_rotation_closed(turns: int) -> None:
    """Rotation returns to 0 modulo τ at phase=1.0."""
    rot = periodic_rotation(1.0, turns=turns)
    assert math.isclose(math.fmod(rot, math.tau), 0.0, abs_tol=1e-9)


def test_periodic_rotation_one_turn_quarter() -> None:
    assert math.isclose(periodic_rotation(0.25, turns=1), math.pi / 2.0, rel_tol=1e-9)


@pytest.mark.parametrize("cycles", [1, 2, 3])
def test_periodic_color_shift_closed(cycles: int) -> None:
    """Color shift closes: frac(cycles × 1.0) == 0.0."""
    assert math.isclose(periodic_color_shift(1.0, cycles=cycles), 0.0, abs_tol=1e-12)


def test_periodic_color_shift_zero() -> None:
    assert periodic_color_shift(0.0, cycles=3) == 0.0


def test_circular_orbit_closes() -> None:
    """Position at phase=1.0 returns exactly to phase=0.0."""
    x0, y0 = circular_orbit(0.0, cx=0.1, cy=0.2, radius=0.5)
    x1, y1 = circular_orbit(1.0, cx=0.1, cy=0.2, radius=0.5)
    assert math.isclose(x0, x1, abs_tol=1e-9)
    assert math.isclose(y0, y1, abs_tol=1e-9)


def test_circular_orbit_radius() -> None:
    """Points on the orbit maintain the requested radius from centre."""
    cx, cy, r = 0.0, 0.0, 0.3
    for phase in [0.0, 0.25, 0.5, 0.75]:
        x, y = circular_orbit(phase, cx=cx, cy=cy, radius=r)
        dist = math.hypot(x - cx, y - cy)
        assert math.isclose(dist, r, rel_tol=1e-9), f"phase={phase}: dist={dist}"


@pytest.mark.parametrize("pulses", [1, 2, 3])
def test_periodic_pulse_closed(pulses: int) -> None:
    """Pulse at phase=0.0 and phase=1.0 are equal (both at cos=1 → value=1)."""
    assert math.isclose(periodic_pulse(0.0, pulses=pulses), 1.0, abs_tol=1e-12)
    assert math.isclose(periodic_pulse(1.0, pulses=pulses), 1.0, abs_tol=1e-12)


def test_periodic_pulse_range() -> None:
    """Pulse values stay in [0, 1]."""
    for i in range(100):
        p = i / 100.0
        v = periodic_pulse(p, pulses=2)
        assert 0.0 <= v <= 1.0


def test_smooth_periodic_envelope_closed() -> None:
    """Envelope returns to bias at phase=0.0 and phase=1.0 (sin=0)."""
    v0 = smooth_periodic_envelope(0.0, cycles=1, bias=0.5, amplitude=0.3)
    v1 = smooth_periodic_envelope(1.0, cycles=1, bias=0.5, amplitude=0.3)
    assert math.isclose(v0, 0.5, abs_tol=1e-12)
    assert math.isclose(v1, 0.5, abs_tol=1e-12)
