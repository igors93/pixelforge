"""Stable enumerated types for artwork visual traits.

Using enums for well-known concepts like SymmetryMode prevents generator code
from passing arbitrary strings and makes the legal values discoverable at the
definition site instead of in documentation comments.
"""

from __future__ import annotations

from enum import StrEnum


class SymmetryMode(StrEnum):
    """How the composition relates to a central axis."""

    NONE = "none"
    MIRROR_H = "mirror-h"
    MIRROR_V = "mirror-v"
    RADIAL = "radial"
    BROKEN = "broken"


class ComplexityLevel(StrEnum):
    """Coarse control over the number of competing visual elements."""

    MINIMAL = "minimal"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    INTRICATE = "intricate"


class DetailLevel(StrEnum):
    """Control over fine-grained detail and iteration depth."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LightingMode(StrEnum):
    """How brightness and contrast are distributed across the composition."""

    FLAT = "flat"
    RADIAL = "radial"
    DIRECTIONAL = "directional"
    AMBIENT = "ambient"


class BackgroundMode(StrEnum):
    """Luminance and treatment of the compositional background."""

    DARK = "dark"
    LIGHT = "light"
    GRADIENT = "gradient"
    VOID = "void"


class AccentMode(StrEnum):
    """Secondary visual emphasis applied over the base composition."""

    NONE = "none"
    HIGHLIGHTS = "highlights"
    SPARKS = "sparks"
    LUMINOUS = "luminous"
