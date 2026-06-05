"""Name-based registry for procedural generators."""

from collections.abc import Iterable

from pixel_forge.core.exceptions import DuplicateGeneratorError, GeneratorNotFoundError
from pixel_forge.core.protocols import ImageGenerator


class GeneratorRegistry:
    """Store generators behind stable names and provide controlled lookup."""

    def __init__(self, generators: Iterable[ImageGenerator] = ()) -> None:
        self._generators: dict[str, ImageGenerator] = {}
        for generator in generators:
            self.register(generator)

    def register(self, generator: ImageGenerator) -> None:
        normalized_name = self._normalize_name(generator.name)
        if normalized_name in self._generators:
            raise DuplicateGeneratorError(
                f"A generator named '{normalized_name}' is already registered."
            )
        self._generators[normalized_name] = generator

    def get(self, name: str) -> ImageGenerator:
        normalized_name = self._normalize_name(name)
        try:
            return self._generators[normalized_name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise GeneratorNotFoundError(
                f"Unknown generator '{name}'. Available generators: {available}."
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._generators))

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()


def build_default_registry() -> GeneratorRegistry:
    """Build the production registry in one explicit composition location."""

    # Local imports keep the composition root explicit and help avoid circular
    # dependencies between the registry and generator packages.
    from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
    from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
    from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
    from pixel_forge.generators.radial_bloom import RadialBloomGenerator
    from pixel_forge.generators.random_noise import RandomNoiseGenerator

    return GeneratorRegistry(
        [
            HarmonicWavesGenerator(),
            PlasmaFlowGenerator(),
            RadialBloomGenerator(),
            MandelbrotDreamGenerator(),
            RandomNoiseGenerator(),
        ]
    )
