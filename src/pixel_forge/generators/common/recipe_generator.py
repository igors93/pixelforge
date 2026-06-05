"""Protocol for recipe-driven generators.

A recipe-driven generator separates two concerns:

1. ``build_recipe`` – make all random decisions and produce an ``ArtworkRecipe``.
2. ``render_recipe`` – render pixels deterministically from a complete recipe.

This separation guarantees that:
  * Compatibility rules and quality evaluation can inspect decisions before
    rendering begins (no pixels allocated until the recipe is accepted).
  * A saved JSON manifest can be replayed: deserialize the recipe, call
    ``render_recipe``, and obtain the same image.
  * The renderer is a pure function of the recipe – it never calls the RNG.

The ``@runtime_checkable`` decorator allows ``isinstance`` checks in the
service layer without requiring explicit class registration.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.core.models.generation_request import GenerationRequest
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.trait_probability import TraitProbability


@runtime_checkable
class RecipeGenerator(Protocol):
    """Contract for generators that expose the recipe-driven pipeline."""

    @property
    def name(self) -> str:
        """Stable public generator name."""
        ...

    def build_recipe(
        self,
        request: GenerationRequest,
        streams: RandomStreams,
        candidate_seed: int,
        retry_index: int,
    ) -> tuple[ArtworkRecipe, list[TraitProbability]]:
        """Make all random decisions and return (recipe, trait_probabilities).

        Must use *streams* for all randomness. Must not allocate pixel arrays.
        The returned trait list is consumed by RarityEvaluator.
        """
        ...

    def render_recipe(self, recipe: ArtworkRecipe) -> UInt8Array:
        """Render a uint8 RGB array from a complete, validated recipe.

        Must be deterministic given the recipe. Must not sample from any RNG.
        """
        ...
