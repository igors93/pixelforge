import pytest

from pixel_forge.core.exceptions import DuplicateGeneratorError, GeneratorNotFoundError
from pixel_forge.generators.random_noise import RandomNoiseGenerator
from pixel_forge.generators.registry import GeneratorRegistry


def test_registry_normalizes_generator_names() -> None:
    registry = GeneratorRegistry([RandomNoiseGenerator()])

    assert registry.get("  RANDOM-NOISE  ").name == "random-noise"


def test_registry_rejects_duplicate_names() -> None:
    registry = GeneratorRegistry([RandomNoiseGenerator()])

    with pytest.raises(DuplicateGeneratorError):
        registry.register(RandomNoiseGenerator())


def test_registry_reports_unknown_generator() -> None:
    registry = GeneratorRegistry([RandomNoiseGenerator()])

    with pytest.raises(GeneratorNotFoundError, match="random-noise"):
        registry.get("missing")
