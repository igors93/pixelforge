"""Tests for the aggregate rarity evaluator.

The overall tier is now determined by total information bits, not just the single
rarest trait. This rewards recipes that combine multiple uncommon traits.

Aggregate thresholds (from rarity_evaluator.py):
  Common:    total_bits < 15
  Uncommon:  total_bits < 25
  Rare:      total_bits < 40
  Epic:      total_bits < 60
  Legendary: total_bits >= 60
"""

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


def test_single_common_trait_is_common() -> None:
    evaluator = RarityEvaluator()
    # 0.24 → -log2(0.24) ≈ 2.06 bits → Common (< 15)
    result = evaluator.evaluate([_tp("petals", "6", 0.24)])
    assert result.overall_tier == RarityTier.COMMON.value


def test_many_traits_combine_to_higher_tier() -> None:
    evaluator = RarityEvaluator()
    # 7 traits each at p=0.001 → 7 × (-log2(0.001)) ≈ 7 × 9.97 = 69.8 bits → Legendary
    traits = [_tp(f"t{i}", "v", 0.001) for i in range(7)]
    result = evaluator.evaluate(traits)
    assert result.overall_tier == RarityTier.LEGENDARY.value


def test_single_rare_event_alone_is_common_by_total_bits() -> None:
    evaluator = RarityEvaluator()
    # p=0.001 → 9.97 bits < 15 → Common (total-bits classification)
    result = evaluator.evaluate([_tp("event", "singularity", 0.001)])
    assert result.overall_tier == RarityTier.COMMON.value


def test_uncommon_threshold_is_between_15_and_25_bits() -> None:
    evaluator = RarityEvaluator()
    # 5 traits each at p=0.10 → 5 × 3.32 = 16.6 bits → Uncommon
    traits = [_tp(f"t{i}", "v", 0.10) for i in range(5)]
    result = evaluator.evaluate(traits)
    assert result.overall_tier == RarityTier.UNCOMMON.value


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


def test_duplicate_trait_names_are_deduplicated() -> None:
    evaluator = RarityEvaluator()
    # Two traits with the same name should not appear twice in trait_details.
    traits = [_tp("same", "a", 0.5), _tp("same", "b", 0.3)]
    result = evaluator.evaluate(traits)
    assert "same" in result.trait_details
    # The total bits should still accumulate correctly.
    assert result.total_information_bits > 0.0
