"""Public exception types raised by PixelForge."""

from pixel_forge.core.exceptions.errors import (
    DuplicateGeneratorError,
    GeneratorNotFoundError,
    IncompatibleTraitsError,
    OutputFileExistsError,
    OutputWriteError,
    PaletteNotFoundError,
    PixelForgeError,
    QualityRejectionError,
    UnsupportedOutputFormatError,
    ValidationError,
)

__all__ = [
    "DuplicateGeneratorError",
    "GeneratorNotFoundError",
    "IncompatibleTraitsError",
    "OutputFileExistsError",
    "OutputWriteError",
    "PaletteNotFoundError",
    "PixelForgeError",
    "QualityRejectionError",
    "UnsupportedOutputFormatError",
    "ValidationError",
]
