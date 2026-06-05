from pathlib import Path

import pytest

from pixel_forge.core.config import Settings
from pixel_forge.core.exceptions import ValidationError
from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.shared.validation import RequestValidator


def make_request(width: int, height: int, seed: int | None = None) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name="random-noise",
        output_path=Path("output/test.png"),
        seed=seed,
    )


def test_accepts_dimensions_at_the_configured_limits() -> None:
    settings = Settings()
    validator = RequestValidator(settings)

    validator.validate(make_request(settings.min_width, settings.max_height))


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 100), (1001, 100), (100, 0), (100, 1001)],
)
def test_rejects_dimensions_outside_the_limits(width: int, height: int) -> None:
    validator = RequestValidator(Settings())

    with pytest.raises(ValidationError):
        validator.validate(make_request(width, height))


def test_rejects_negative_seed() -> None:
    validator = RequestValidator(Settings())

    with pytest.raises(ValidationError, match="Seed"):
        validator.validate(make_request(10, 10, seed=-1))
