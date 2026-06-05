"""Named palette system for perceptually controlled color mapping."""

from pixel_forge.aesthetics.palettes.color_palette import ColorPalette
from pixel_forge.aesthetics.palettes.default_palettes import DEFAULT_PALETTES
from pixel_forge.aesthetics.palettes.palette_registry import (
    PaletteRegistry,
    build_default_palette_registry,
)

__all__ = [
    "ColorPalette",
    "DEFAULT_PALETTES",
    "PaletteRegistry",
    "build_default_palette_registry",
]
