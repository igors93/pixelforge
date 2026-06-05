"""Centralized application settings.

Keeping operational limits in one immutable object prevents values from being
scattered across the CLI, validators, and generators.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime defaults and safety limits for image generation."""

    min_width: int = 1
    min_height: int = 1
    max_width: int = 1000
    max_height: int = 1000
    default_width: int = 256
    default_height: int = 256
    default_generator: str = "random-noise"
    default_output_path: Path = Path("output/random-noise.png")
    supported_output_suffixes: tuple[str, ...] = (".png",)
    png_compress_level: int = 6
