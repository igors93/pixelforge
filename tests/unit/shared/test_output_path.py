from pathlib import Path

import pytest

from pixel_forge.core.exceptions import UnsupportedOutputFormatError
from pixel_forge.shared.paths import normalize_output_path


def test_adds_png_suffix_when_missing() -> None:
    result = normalize_output_path(
        Path("output/image"),
        supported_suffixes=(".png",),
    )

    assert result == Path("output/image.png")


def test_rejects_unsupported_suffix() -> None:
    with pytest.raises(UnsupportedOutputFormatError):
        normalize_output_path(
            Path("output/image.jpg"),
            supported_suffixes=(".png",),
        )
