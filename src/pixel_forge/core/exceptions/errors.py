"""Domain-specific exceptions used for predictable error handling."""


class PixelForgeError(Exception):
    """Base class for errors that can be safely presented to CLI users."""


class ValidationError(PixelForgeError):
    """Raised when a generation request violates an application rule."""


class GeneratorNotFoundError(PixelForgeError):
    """Raised when a requested generator is not registered."""


class DuplicateGeneratorError(PixelForgeError):
    """Raised when two generators use the same public name."""


class UnsupportedOutputFormatError(PixelForgeError):
    """Raised when the output file extension is not supported."""


class OutputFileExistsError(PixelForgeError):
    """Raised when an output file exists and overwrite was not requested."""


class OutputWriteError(PixelForgeError):
    """Raised when encoded image bytes cannot be written to storage."""
