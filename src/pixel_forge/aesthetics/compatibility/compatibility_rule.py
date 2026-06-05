"""Protocol and result model for compatibility rules.

Compatibility rules inspect a recipe and either return it unmodified or return
a modified copy alongside the rule name that was applied. Rules must be pure,
deterministic functions – they must never introduce new random decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pixel_forge.core.models.artwork_recipe import ArtworkRecipe


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """A recipe (possibly modified) and the names of rules that were applied."""

    recipe: ArtworkRecipe
    applied_rules: tuple[str, ...]


class CompatibilityRule(Protocol):
    """A deterministic function that may modify a recipe for aesthetic safety."""

    @property
    def name(self) -> str:
        """Short rule identifier recorded in the manifest."""
        ...

    def apply(self, recipe: ArtworkRecipe) -> ArtworkRecipe:
        """Return a (possibly modified) recipe.

        Return the same object when no modification is needed.
        Must not perform any random sampling.
        """
        ...
