"""Validation of GenerationOptions supplied by the user.

All validation raises OptionsValidationError with a descriptive message so the
CLI can present the error without a traceback. No assert statements are used —
assert is only for invariant checks in test code, not operational validation.
"""

from __future__ import annotations

from pixel_forge.core.exceptions.errors import OptionsValidationError
from pixel_forge.core.models.generation_options import GenerationOptions

_VALID_COMPLEXITY_VALUES = frozenset(
    ["minimal", "simple", "moderate", "complex", "intricate"]
)
_VALID_RARITY_TIERS = frozenset(
    ["Common", "Uncommon", "Rare", "Epic", "Legendary"]
)
_MAX_ALLOWED_RETRIES = 50


def validate_generation_options(
    options: GenerationOptions,
    *,
    palette_registry: object | None = None,
) -> None:
    """Validate all fields of *options* and raise OptionsValidationError on failure.

    Pass *palette_registry* to enable palette existence checking.
    """
    errors: list[str] = []

    if (
        options.quality_threshold is not None
        and not (0.0 <= options.quality_threshold <= 1.0)
    ):
        errors.append(
            f"--quality-threshold must be in [0.0, 1.0]; "
            f"got {options.quality_threshold!r}."
        )

    if options.max_retries < 0:
        errors.append(
            f"--max-retries must be non-negative; got {options.max_retries!r}."
        )
    if options.max_retries > _MAX_ALLOWED_RETRIES:
        errors.append(
            f"--max-retries must not exceed {_MAX_ALLOWED_RETRIES}; "
            f"got {options.max_retries!r}."
        )

    if (
        options.complexity_level is not None
        and options.complexity_level not in _VALID_COMPLEXITY_VALUES
    ):
        errors.append(
            f"--complexity must be one of "
            f"{sorted(_VALID_COMPLEXITY_VALUES)}; "
            f"got {options.complexity_level!r}."
        )

    if (
        options.min_rarity_tier is not None
        and options.min_rarity_tier not in _VALID_RARITY_TIERS
    ):
        errors.append(
            f"--min-rarity must be one of {sorted(_VALID_RARITY_TIERS)}; "
            f"got {options.min_rarity_tier!r}."
        )

    if options.palette_name is not None and palette_registry is not None:
        from pixel_forge.aesthetics.palettes.palette_registry import PaletteRegistry

        if isinstance(palette_registry, PaletteRegistry):
            available = set(palette_registry.names())
            if options.palette_name not in available:
                errors.append(
                    f"--palette '{options.palette_name}' is not a known palette. "
                    f"Available: {sorted(available)}."
                )

    if errors:
        raise OptionsValidationError(
            "Invalid generation options:\n" + "\n".join(f"  • {e}" for e in errors)
        )
