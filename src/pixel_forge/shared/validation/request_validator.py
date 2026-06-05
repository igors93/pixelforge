"""Validation rules for generation requests."""

from pixel_forge.core.config import Settings
from pixel_forge.core.exceptions import ValidationError
from pixel_forge.core.models import GenerationRequest


class RequestValidator:
    """Validate input at the application boundary before expensive work begins."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(self, request: GenerationRequest) -> None:
        self._validate_dimension(
            name="width",
            value=request.size.width,
            minimum=self._settings.min_width,
            maximum=self._settings.max_width,
        )
        self._validate_dimension(
            name="height",
            value=request.size.height,
            minimum=self._settings.min_height,
            maximum=self._settings.max_height,
        )

        if not request.generator_name.strip():
            raise ValidationError("Generator name cannot be empty.")

        if request.seed is not None and request.seed < 0:
            raise ValidationError("Seed must be zero or a positive integer.")

    @staticmethod
    def _validate_dimension(*, name: str, value: int, minimum: int, maximum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"Image {name} must be an integer.")
        if not minimum <= value <= maximum:
            raise ValidationError(
                f"Image {name} must be between {minimum} and {maximum} pixels; received {value}."
            )
