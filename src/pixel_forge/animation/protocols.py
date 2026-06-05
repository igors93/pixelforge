"""Protocol for animated recipe renderers.

An AnimatedRecipeRenderer separates the same two concerns as RecipeGenerator,
extended to animation:

1. ``build_animation_recipe`` – sample all motion decisions from AnimationStreams
   and return an immutable AnimationRecipe. No pixels are allocated here.

2. ``render_frame`` – render a single uint8 RGB frame deterministically from
   the AnimationRecipe and a normalized phase value. No RNG calls allowed.

The phase value is the only source of time-varying information. The renderer
must produce a periodic output: render_frame(recipe, 0.0) must be
mathematically identical to the limit of render_frame(recipe, phase) as
phase → 1.0 (i.e. the virtual frame at phase=1.0 equals phase=0.0).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pixel_forge.animation.animation_randomness import AnimationStreams
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import AnimationRecipe
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.rarity.trait_probability import TraitProbability


@runtime_checkable
class AnimatedRecipeRenderer(Protocol):
    """Contract for generators that support animated GIF output."""

    @property
    def name(self) -> str:
        """Stable public generator name (must match RecipeGenerator.name)."""
        ...

    def build_animation_recipe(
        self,
        artwork_recipe: ArtworkRecipe,
        options: AnimationOptions,
        streams: AnimationStreams,
        animation_seed: int,
    ) -> tuple[AnimationRecipe, list[TraitProbability]]:
        """Sample all motion decisions and return (animation_recipe, trait_probs).

        Must use *streams* for all randomness.  Must not allocate pixel arrays.
        The returned trait list feeds the animation rarity evaluator.
        """
        ...

    def render_frame(
        self,
        animation_recipe: AnimationRecipe,
        phase: float,
    ) -> UInt8Array:
        """Render a single uint8 RGB frame at the given *phase* ∈ [0.0, 1.0).

        Must be deterministic — no RNG calls, no side effects, no file I/O.
        Must satisfy render_frame(recipe, 0.0) == render_frame(recipe, 0.0)
        and the path must be periodic: virtual phase 1.0 == actual phase 0.0.
        """
        ...
