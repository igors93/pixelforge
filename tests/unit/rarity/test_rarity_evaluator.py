"""Tests for the aggregate rarity evaluator."""

from __future__ import annotations

import math

import pytest

from pixel_forge.rarity.rarity_evaluator import RarityEvaluator
from pixel_forge.rarity.rarity_tier import RarityTier
from pixel_forge.rarity.trait_probability import TraitProbability


def _tp(name: str, value: str, probability: float) -> TraitProbability:
    return TraitProbability(trait_name=name, value=value, probability=probability)


def test_empty_traits_return_common_tier() -> None:
    evaluator = RarityEvaluator()
    result = evaluator.evaluate([])
    assert result.overall_tier == RarityTier.COMMON.value


def test_single_common_trait() -> None:
    evaluator = RarityEvaluator()
    result = evaluator.evaluate([_tp("petals", "6", 0.24)])
    assert result.overall_tier == RarityTier.COMMON.value


def test_single_legendary_trait() -> None:
    evaluator = RarityEvaluator()
    result = evaluator.evaluate([_tp("event", "singularity", 0.001)])
    assert result.overall_tier == RarityTier.LEGENDARY.value


def test_information_bits_are_positive() -> None:
    evaluator = RarityEvaluator()
    result = evaluator.evaluate([_tp("x", "a", 0.5), _tp("y", "b", 0.1)])
    assert result.total_information_bits > 0.0


def test_total_bits_equals_sum_of_individual_bits() -> None:
    evaluator = RarityEvaluator()
    traits = [_tp("a", "v", 0.5), _tp("b", "w", 0.1), _tp("c", "x", 0.02)]
    result = evaluator.evaluate(traits)
    expected = sum(-math.log2(t.probability) for t in traits)
    assert result.total_information_bits == pytest.approx(expected, rel=1e-6)


def test_most_significant_traits_are_sorted_descending() -> None:
    evaluator = RarityEvaluator()
    traits = [
        _tp("common", "a", 0.5),
        _tp("epic", "b", 0.004),
        _tp("uncommon", "c", 0.1),
    ]
    result = evaluator.evaluate(traits)
    bits = [e.information_bits for e in result.most_significant_traits]
    assert bits == sorted(bits, reverse=True)


def test_trait_details_contain_all_traits() -> None:
    evaluator = RarityEvaluator()
    traits = [_tp("a", "1", 0.5), _tp("b", "2", 0.2), _tp("c", "3", 0.01)]
    result = evaluator.evaluate(traits)
    assert set(result.trait_details.keys()) == {"a", "b", "c"}


def test_summary_is_non_empty_string() -> None:
    evaluator = RarityEvaluator()
    result = evaluator.evaluate([_tp("x", "v", 0.2)])
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
