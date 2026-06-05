"""Output path normalization rules."""

from pathlib import Path

from pixel_forge.core.exceptions import UnsupportedOutputFormatError, ValidationError


def normalize_output_path(
    output_path: Path,
    *,
    supported_suffixes: tuple[str, ...],
) -> Path:
    """Normalize a user path and enforce a supported file extension."""

    path = output_path.expanduser()

    if not path.name or path.name in {".", ".."}:
        raise ValidationError("Output path must include a file name.")

    if not path.suffix:
        path = path.with_suffix(supported_suffixes[0])

    normalized_suffix = path.suffix.lower()
    if normalized_suffix not in supported_suffixes:
        supported = ", ".join(supported_suffixes)
        raise UnsupportedOutputFormatError(
            f"Unsupported output extension '{path.suffix}'. Supported extensions: {supported}."
        )

    return path.with_suffix(normalized_suffix)
