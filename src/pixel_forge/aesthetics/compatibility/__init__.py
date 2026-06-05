"""Compatibility rules that constrain recipes before rendering begins."""

from pixel_forge.aesthetics.compatibility.compatibility_rule import CompatibilityResult
from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)

__all__ = ["CompatibilityResult", "RecipeCompatibilityValidator"]
