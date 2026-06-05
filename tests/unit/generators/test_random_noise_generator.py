from pathlib import Path

from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.generators.random_noise import RandomNoiseGenerator


def build_request(seed: int) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=4, height=3),
        generator_name="random-noise",
        output_path=Path("unused.png"),
        seed=seed,
    )


def test_seeded_generation_is_reproducible() -> None:
    generator = RandomNoiseGenerator()

    first = generator.generate(build_request(seed=123))
    second = generator.generate(build_request(seed=123))

    assert first.pixels == second.pixels
    assert first.seed == 123
    assert len(first.pixels) == 4 * 3 * 3


def test_different_seeds_produce_different_pixel_data() -> None:
    generator = RandomNoiseGenerator()

    first = generator.generate(build_request(seed=1))
    second = generator.generate(build_request(seed=2))

    assert first.pixels != second.pixels
