"""Tests for frame phase generation."""

from __future__ import annotations

import pytest

from pixel_forge.animation.frame_phase import generate_frame_phases, representative_phases


def test_phase_count_matches_frame_count() -> None:
    phases = generate_frame_phases(12)
    assert len(phases) == 12


def test_phases_start_at_zero() -> None:
    phases = generate_frame_phases(8)
    assert phases[0] == 0.0


def test_phases_end_before_one() -> None:
    phases = generate_frame_phases(8)
    assert phases[-1] < 1.0


def test_no_duplicate_endpoint() -> None:
    phases = generate_frame_phases(24)
    assert 1.0 not in phases


def test_phases_evenly_spaced() -> None:
    n = 10
    phases = generate_frame_phases(n)
    expected = [i / n for i in range(n)]
    for got, exp in zip(phases, expected):
        assert abs(got - exp) < 1e-12


@pytest.mark.parametrize("n", [2, 4, 12, 24, 48, 60])
def test_phases_valid_range(n: int) -> None:
    phases = generate_frame_phases(n)
    assert all(0.0 <= p < 1.0 for p in phases)


def test_frame_count_1_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        generate_frame_phases(1)


def test_representative_phases_has_four() -> None:
    phases = representative_phases()
    assert len(phases) == 4


def test_representative_phases_values() -> None:
    phases = representative_phases()
    assert phases == [0.0, 0.25, 0.5, 0.75]
