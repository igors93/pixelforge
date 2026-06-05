"""Registry for named color palettes with weighted random selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pixel_forge.aesthetics.palettes.color_palette import ColorPalette
from pixel_forge.aesthetics.palettes.default_palettes import DEFAULT_PALETTES
from pixel_forge.core.exceptions import PaletteNotFoundError
from pixel_forge.randomness.weighted_choice import WeightedChoice, sample_weighted


class PaletteRegistry:
    """Store named palettes and support weighted random selection."""

    def __init__(self, palettes: Iterable[ColorPalette] = ()) -> None:
        self._palettes: dict[str, ColorPalette] = {}
        for palette in palettes:
            self.register(palette)

    def register(self, palette: ColorPalette) -> None:
        self._palettes[palette.name] = palette

    def get(self, name: str) -> ColorPalette:
        try:
            return self._palettes[name]
        except KeyError as error:
            available = ", ".join(sorted(self._palettes)) or "none"
            raise PaletteNotFoundError(
                f"Unknown palette '{name}'. Available palettes: {available}."
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._palettes))

    def sample(
        self,
        rng: np.random.Generator,
        *,
        compatible_generator: str | None = None,
    ) -> ColorPalette:
        """Draw one palette proportionally to its sampling_probability.

        When *compatible_generator* is provided, palettes whose
        compatible_generators tuple is non-empty are filtered to only those
        that list the requested generator. Palettes with an empty
        compatible_generators tuple are always eligible.
        """
        candidates = [
            p for p in self._palettes.values()
            if not p.compatible_generators
            or (
                compatible_generator is not None
                and compatible_generator in p.compatible_generators
            )
        ]
        if not candidates:
            candidates = list(self._palettes.values())

        choices = [WeightedChoice(value=p, weight=p.sampling_probability) for p in candidates]
        result = sample_weighted(choices, rng)
        return result.value


def build_default_palette_registry() -> PaletteRegistry:
    return PaletteRegistry(DEFAULT_PALETTES)
