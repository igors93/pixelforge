"""Tests for rarity tier thresholds and probability classification."""

from __future__ import annotations

import pytest

from pixel_forge.rarity.rarity_tier import RarityTier, tier_for_probability


@pytest.mark.parametrize(
    ("probability", "expected_tier"),
    [
        (1.00, RarityTier.COMMON),
        (0.50, RarityTier.COMMON),
        (0.20, RarityTier.COMMON),
        (0.199, RarityTier.UNCOMMON),
        (0.08, RarityTier.UNCOMMON),
        (0.079, RarityTier.RARE),
        (0.02, RarityTier.RARE),
        (0.019, RarityTier.EPIC),
        (0.005, RarityTier.EPIC),
        (0.004, RarityTier.LEGENDARY),
        (0.001, RarityTier.LEGENDARY),
        (0.0001, RarityTier.LEGENDARY),
        (0.0, RarityTier.LEGENDARY),
    ],
)
def test_tier_for_probability(probability: float, expected_tier: RarityTier) -> None:
    assert tier_for_probability(probability) == expected_tier


def test_rarity_tiers_are_strings() -> None:
    for tier in RarityTier:
        assert isinstance(tier.value, str)
        assert tier.value  # not empty


def test_all_tier_values_distinct() -> None:
    values = [t.value for t in RarityTier]
    assert len(values) == len(set(values))
