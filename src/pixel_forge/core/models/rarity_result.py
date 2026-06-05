"""Result model for aggregate rarity evaluation of a complete recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TraitRarityEntry:
    """Rarity information for one sampled trait."""

    trait_name: str
    value: str           # string representation of the sampled value
    probability: float   # normalized probability of this value in its distribution
    information_bits: float  # -log2(probability) – higher means rarer


@dataclass(frozen=True, slots=True)
class RarityResult:
    """Aggregate rarity evaluation for a complete artwork recipe.

    Rarity is descriptive: it reflects the mathematical probability of the
    sampled combination, not a guarantee of visual uniqueness or objective value.
    """

    overall_tier: str            # e.g. "Common", "Uncommon", "Rare", "Epic", "Legendary"
    total_information_bits: float  # sum of information_bits across all traits
    most_significant_traits: tuple[TraitRarityEntry, ...]  # sorted by information_bits desc
    trait_details: Mapping[str, TraitRarityEntry] = field(hash=False)
    summary: str = ""            # human-readable description
