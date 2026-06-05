"""Tests for independent deterministic random streams."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_forge.randomness.random_streams import RandomStreams


def test_streams_from_same_seed_are_identical() -> None:
    a = RandomStreams.from_seed(42)
    b = RandomStreams.from_seed(42)
    # Draw one value from each named stream and compare.
    assert float(a.traits.random()) == pytest.approx(float(b.traits.random()))
    assert float(a.geometry.random()) == pytest.approx(float(b.geometry.random()))
    assert float(a.palette.random()) == pytest.approx(float(b.palette.random()))


def test_streams_from_different_seeds_differ() -> None:
    a = RandomStreams.from_seed(1)
    b = RandomStreams.from_seed(2)
    v_a = float(a.traits.random())
    v_b = float(b.traits.random())
    assert v_a != pytest.approx(v_b)


def test_streams_are_independent() -> None:
    """Drawing from one stream must not advance another stream's state."""
    streams_a = RandomStreams.from_seed(99)
    streams_b = RandomStreams.from_seed(99)

    # Advance the traits stream on streams_a many times.
    for _ in range(100):
        streams_a.traits.random()

    # The geometry stream on both should still agree.
    v_a = float(streams_a.geometry.random())
    v_b = float(streams_b.geometry.random())
    assert v_a == pytest.approx(v_b)


def test_all_named_streams_are_present() -> None:
    streams = RandomStreams.from_seed(0)
    for attr in (
        "traits",
        "geometry",
        "composition",
        "palette",
        "lighting",
        "accents",
        "rarity",
        "quality_retry",
    ):
        assert hasattr(streams, attr)
        assert isinstance(getattr(streams, attr), np.random.Generator)


def test_streams_produce_valid_floats() -> None:
    streams = RandomStreams.from_seed(7)
    for rng in (
        streams.traits,
        streams.geometry,
        streams.composition,
        streams.palette,
        streams.lighting,
        streams.accents,
        streams.rarity,
        streams.quality_retry,
    ):
        v = float(rng.random())
        assert 0.0 <= v < 1.0
