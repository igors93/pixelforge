"""Aggregate rarity evaluator for a complete set of sampled traits.

Information bits are summed rather than multiplying probabilities together to
avoid floating-point underflow when many low-probability traits are combined
in a single recipe. Summing logarithms is numerically equivalent to the log
of the joint probability.
"""

from __future__ import annotations

from pixel_forge.core.models.rarity_result import RarityResult, TraitRarityEntry
from pixel_forge.rarity.rarity_tier import RarityTier, tier_for_probability
from pixel_forge.rarity.trait_probability import TraitProbability

# Number of most-significant (rarest) traits to include in the summary list.
_TOP_TRAITS_COUNT = 5


class RarityEvaluator:
    """Evaluate aggregate rarity from a collection of trait probabilities."""

    def evaluate(self, traits: list[TraitProbability]) -> RarityResult:
        """Compute a RarityResult from the given trait probability records."""
        if not traits:
            return self._empty_result()

        entries = {
            tp.trait_name: TraitRarityEntry(
                trait_name=tp.trait_name,
                value=tp.value,
                probability=tp.probability,
                information_bits=tp.information_bits,
            )
            for tp in traits
        }

        total_bits = sum(e.information_bits for e in entries.values())

        sorted_entries = sorted(
            entries.values(),
            key=lambda e: e.information_bits,
            reverse=True,
        )
        top_traits = tuple(sorted_entries[:_TOP_TRAITS_COUNT])

        # Overall tier is determined by the single rarest trait's probability.
        rarest_probability = sorted_entries[0].probability if sorted_entries else 1.0
        overall_tier = tier_for_probability(rarest_probability)

        summary = self._build_summary(overall_tier, total_bits, top_traits)

        return RarityResult(
            overall_tier=overall_tier.value,
            total_information_bits=total_bits,
            most_significant_traits=top_traits,
            trait_details=entries,
            summary=summary,
        )

    @staticmethod
    def _empty_result() -> RarityResult:
        tier = RarityTier.COMMON
        return RarityResult(
            overall_tier=tier.value,
            total_information_bits=0.0,
            most_significant_traits=(),
            trait_details={},
            summary="No sampled traits recorded.",
        )

    @staticmethod
    def _build_summary(
        tier: RarityTier,
        total_bits: float,
        top_traits: tuple[TraitRarityEntry, ...],
    ) -> str:
        parts = [f"Overall: {tier.value} ({total_bits:.1f} information bits total)."]
        if top_traits:
            rare_labels = [
                f"{e.trait_name}={e.value} ({e.probability * 100:.2f}%)"
                for e in top_traits
                if tier_for_probability(e.probability) not in (
                    RarityTier.COMMON, RarityTier.UNCOMMON
                )
            ]
            if rare_labels:
                parts.append("Rarest traits: " + ", ".join(rare_labels) + ".")
        return " ".join(parts)
