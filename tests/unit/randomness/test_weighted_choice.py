"""Tests for weighted sampler validation and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_sample_returns_a_value_from_choices() -> None:
    choices = [WeightedChoice(value="a", weight=1.0), WeightedChoice(value="b", weight=1.0)]
    result = sample_weighted(choices, _rng(0))
    assert result.value in ("a", "b")


def test_sample_is_deterministic_for_fixed_seed() -> None:
    choices = [WeightedChoice(value=i, weight=float(i + 1)) for i in range(5)]
    r1 = sample_weighted(choices, _rng(42))
    r2 = sample_weighted(choices, _rng(42))
    assert r1.value == r2.value
    assert r1.probability == pytest.approx(r2.probability)


def test_sample_probability_sums_to_one_conceptually() -> None:
    choices = [WeightedChoice(value=i, weight=float(i + 1)) for i in range(4)]
    total_weight = sum(c.weight for c in choices)
    result = sample_weighted(choices, _rng(7))
    expected_prob = choices[result.value].weight / total_weight
    assert result.probability == pytest.approx(expected_prob, abs=1e-10)


def test_empty_choices_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        sample_weighted([], _rng(0))


def test_negative_weight_raises() -> None:
    choices = [WeightedChoice(value="a", weight=1.0), WeightedChoice(value="b", weight=-0.5)]
    with pytest.raises(ValueError, match="non-negative"):
        sample_weighted(choices, _rng(0))


def test_zero_total_weight_raises() -> None:
    choices = [WeightedChoice(value="a", weight=0.0), WeightedChoice(value="b", weight=0.0)]
    with pytest.raises(ValueError, match="greater than zero"):
        sample_weighted(choices, _rng(0))


def test_single_choice_always_selected() -> None:
    choices = [WeightedChoice(value="only", weight=99.9)]
    result = sample_weighted(choices, _rng(12345))
    assert result.value == "only"
    assert result.probability == pytest.approx(1.0)


def test_heavy_choice_sampled_more_often() -> None:
    choices = [
        WeightedChoice(value="rare", weight=1.0),
        WeightedChoice(value="common", weight=99.0),
    ]
    counts: dict[str, int] = {"rare": 0, "common": 0}
    rng = _rng(0)
    for _ in range(200):
        r = sample_weighted(choices, rng)
        counts[r.value] += 1
    # With a 1:99 ratio, "common" should win overwhelmingly.
    assert counts["common"] > counts["rare"] * 5


def test_sampling_result_is_frozen() -> None:
    choices = [WeightedChoice(value="x", weight=1.0)]
    result = sample_weighted(choices, _rng(0))
    with pytest.raises((AttributeError, TypeError)):
        result.value = "y"  # type: ignore[misc]
