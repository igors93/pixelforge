from pathlib import Path

import pytest

from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.core.protocols import ImageGenerator
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
from pixel_forge.generators.radial_bloom import RadialBloomGenerator


@pytest.mark.parametrize(
    ("generator", "generator_name"),
    [
        (HarmonicWavesGenerator(), "harmonic-waves"),
        (PlasmaFlowGenerator(), "plasma-flow"),
        (RadialBloomGenerator(), "radial-bloom"),
        (MandelbrotDreamGenerator(), "mandelbrot-dream"),
    ],
)
def test_artistic_generators_are_reproducible(
    generator: ImageGenerator,
    generator_name: str,
) -> None:
    request = GenerationRequest(
        size=ImageSize(width=32, height=24),
        generator_name=generator_name,
        output_path=Path("unused.png"),
        seed=123,
    )

    first = generator.generate(request)
    second = generator.generate(request)

    assert first.pixels == second.pixels
    assert first.generator_name == generator_name
    assert len(first.pixels) == 32 * 24 * 3
    assert len(set(first.pixels)) > 16


def test_harmonic_waves_generator_changes_output_with_seed() -> None:
    generator = HarmonicWavesGenerator()
    first = generator.generate(
        GenerationRequest(
            size=ImageSize(width=32, height=24),
            generator_name="harmonic-waves",
            output_path=Path("unused.png"),
            seed=1,
        )
    )
    second = generator.generate(
        GenerationRequest(
            size=ImageSize(width=32, height=24),
            generator_name="harmonic-waves",
            output_path=Path("unused.png"),
            seed=2,
        )
    )

    assert first.pixels != second.pixels
