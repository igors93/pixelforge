"""Extended generation options supplied by the user or CLI.

These options override or constrain the sampled recipe values. They are stored
separately from the recipe so that the recipe always reflects the final resolved
state after applying overrides and compatibility rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Optional user-supplied constraints and overrides for generation."""

    palette_name: str | None = None        # force a specific palette
    min_rarity_tier: str | None = None     # "Common" | "Uncommon" | "Rare" | "Epic" | "Legendary"
    complexity_level: str | None = None   # override complexity
    quality_threshold: float | None = None # minimum aggregate quality score [0, 1]
    max_retries: int = 5                   # maximum quality retry attempts
    write_metadata: bool = True            # write JSON manifest beside PNG
    strict_quality: bool = False           # fail rather than return best candidate
