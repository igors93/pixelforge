"""Rarity tier classification and probability thresholds.

All probability thresholds are defined in exactly one location so that
CLI labels, manifest JSON, and test assertions all reference the same values.
"""

from __future__ import annotations

from enum import StrEnum


class RarityTier(StrEnum):
    """Named rarity classification ordered from most to least common."""

    COMMON = "Common"
    UNCOMMON = "Uncommon"
    RARE = "Rare"
    EPIC = "Epic"
    LEGENDARY = "Legendary"


def tier_for_total_bits(total_bits: float) -> RarityTier:
    """Return overall rarity tier from the total information-bit sum.

    Thresholds match those in rarity_evaluator.tier_for_total_bits.
    Defined here so that the service layer can import it without touching
    the evaluator module (avoiding a circular import).

    Common:    total_bits < 15
    Uncommon:  total_bits < 25
    Rare:      total_bits < 40
    Epic:      total_bits < 60
    Legendary: total_bits >= 60
    """
    if total_bits < 15.0:
        return RarityTier.COMMON
    if total_bits < 25.0:
        return RarityTier.UNCOMMON
    if total_bits < 40.0:
        return RarityTier.RARE
    if total_bits < 60.0:
        return RarityTier.EPIC
    return RarityTier.LEGENDARY


def tier_for_probability(probability: float) -> RarityTier:
    """Return the rarity tier for a given normalized probability.

    Thresholds are checked from most common to rarest so that the first
    matching condition (highest threshold first) wins.

    Common:    probability >= 0.20
    Uncommon:  probability >= 0.08
    Rare:      probability >= 0.02
    Epic:      probability >= 0.005
    Legendary: probability < 0.005
    """
    if probability >= 0.20:
        return RarityTier.COMMON
    if probability >= 0.08:
        return RarityTier.UNCOMMON
    if probability >= 0.02:
        return RarityTier.RARE
    if probability >= 0.005:
        return RarityTier.EPIC
    return RarityTier.LEGENDARY
