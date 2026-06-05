import pytest

from pixel_forge.core.exceptions import DuplicateGeneratorError, GeneratorNotFoundError
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.registry import GeneratorRegistry, build_default_registry


def test_registry_normalizes_generator_names() -> None:
    registry = GeneratorRegistry([HarmonicWavesGenerator()])

    assert registry.get("  HARMONIC-WAVES  ").name == "harmonic-waves"


def test_registry_rejects_duplicate_names() -> None:
    registry = GeneratorRegistry([HarmonicWavesGenerator()])

    with pytest.raises(DuplicateGeneratorError):
        registry.register(HarmonicWavesGenerator())


def test_registry_reports_unknown_generator() -> None:
    registry = GeneratorRegistry([HarmonicWavesGenerator()])

    with pytest.raises(GeneratorNotFoundError, match="harmonic-waves"):
        registry.get("missing")


def test_default_registry_contains_artistic_generators() -> None:
    registry = build_default_registry()

    assert registry.names() == (
        "harmonic-waves",
        "mandelbrot-dream",
        "plasma-flow",
        "radial-bloom",
        "random-noise",
    )
