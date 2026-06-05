"""Trait probability record produced during recipe building.

Generators accumulate one TraitProbability per sampled decision and pass the
complete list to the RarityEvaluator. Storing probability alongside the trait
name and value avoids recomputing distributions during evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TraitProbability:
    """Probability metadata for one sampled trait decision."""

    trait_name: str
    value: str            # string representation of the sampled value
    probability: float    # normalized probability in [0, 1]

    @property
    def information_bits(self) -> float:
        """Shannon information content in bits: -log2(probability).

        Higher values mean a rarer selection. A probability of 1.0 → 0 bits
        (certain); a probability of 0.001 → ~10 bits (very rare).
        Log of zero is protected by clamping to a minimum probability.
        """
        p = max(self.probability, 1e-15)
        return -math.log2(p)
