"""Immutable artwork recipe produced before any rendering begins.

A recipe records every visual decision made from the seed and trait samplers.
Storing the recipe separately from the rendered pixels guarantees that:

  * The same recipe always produces the same image (renderer is deterministic).
  * A saved JSON recipe can be replayed without the original seed.
  * Compatibility rules and quality evaluation can inspect decisions before
    pixels are allocated.
  * The manifest can describe the full creative intent of an image.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)

# Bump this string whenever the recipe structure changes in a backward-
# incompatible way so that replayed recipes can be validated against the
# version that produced them.
RECIPE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ArtworkRecipe:
    """Every meaningful visual decision made before a single pixel is rendered.

    Common structural traits are represented as typed enums. Generator-specific
    numeric parameters are collected in ``generator_params``, a flat mapping of
    JSON-serializable primitives populated by each generator's recipe builder.
    The field is excluded from hashing because dict equality already covers it
    in __eq__ and dicts are not hashable.
    """

    schema_version: str
    generator_name: str
    seed: int            # master seed supplied by the user (or derived)
    candidate_seed: int  # effective seed used for this attempt (may differ on retry)
    retry_index: int     # 0 for the first unretried attempt
    width: int
    height: int
    palette_name: str
    symmetry_mode: SymmetryMode
    complexity_level: ComplexityLevel
    detail_level: DetailLevel
    background_mode: BackgroundMode
    lighting_mode: LightingMode
    accent_mode: AccentMode
    rare_events: tuple[str, ...]
    # Flat mapping of generator-specific numeric/string parameters.
    # Values must be JSON-serializable primitives (int, float, str, bool, None).
    generator_params: Mapping[str, Any] = field(hash=False)
