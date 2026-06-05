"""Typed domain models for the geometric-orbit generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Color = tuple[int, int, int]
ShapeKind = Literal[
    "circle",
    "square",
    "triangle",
    "capsule",
    "line",
    "diamond",
    "ring",
]
CompositionStyle = Literal[
    "bubble-field",
    "double-square-crown",
    "triangle-explosion",
    "neon-orbit",
    "radial-capsules",
    "symbolic-minimal",
    "geometric-sun",
    "scattered-constellation",
]
BackgroundStyle = Literal[
    "broad-ribbons",
    "angular-wedges",
    "split-diagonal",
    "crossing-beams",
    "quiet-field",
]
DistributionMode = Literal[
    "single-ring",
    "double-ring",
    "radial-burst",
    "scattered-orbit",
    "constellation",
]
SizeDistribution = Literal[
    "uniform-jitter",
    "multi-scale",
    "wide",
]


@dataclass(frozen=True, slots=True)
class PaletteDefinition:
    """Complete functional color system used by one composition."""

    name: str
    canvas: Color
    panel_primary: Color
    panel_secondary: Color
    panel_accent: Color
    center_fill: Color
    center_outline: Color
    center_highlight: Color
    orbit_fill: Color
    orbit_secondary: Color
    orbit_outline: Color
    inner_fill: Color
    inner_accent: Color
    inner_outline: Color
    shadow: Color


@dataclass(frozen=True, slots=True)
class ShapeGrammar:
    """Rules that give one image a coherent geometric language."""

    name: CompositionStyle
    primary_shape: ShapeKind
    secondary_shape: ShapeKind
    tertiary_shape: ShapeKind | None
    distribution: DistributionMode
    size_distribution: SizeDistribution
    min_count: int
    max_count: int
    ring_count: int
    angular_jitter: float
    radial_jitter: float
    large_shape_probability: float
    secondary_probability: float
    tertiary_probability: float
    allow_outline_only: bool
    background_choices: tuple[BackgroundStyle, ...]


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """One large geometric panel in normalized canvas coordinates."""

    center_x: float
    center_y: float
    width: float
    height: float
    rotation_degrees: float
    color: Color
    corner_radius: float
    shadow_opacity: int
    opacity: int


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    """One clean geometric primitive in normalized coordinates."""

    kind: ShapeKind
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_degrees: float
    fill: Color
    outline: Color
    outline_width: float
    shadow_opacity: int
    fill_opacity: int = 255


@dataclass(frozen=True, slots=True)
class GeometricOrbitParameters:
    """Fully sampled recipe used by the renderer without further randomness."""

    palette: PaletteDefinition
    grammar: ShapeGrammar
    composition_style: CompositionStyle
    background_style: BackgroundStyle
    panels: tuple[PanelSpec, ...]
    outer_shapes: tuple[ShapeSpec, ...]
    inner_shapes: tuple[ShapeSpec, ...]
    center_radius: float
    center_outline_width: float
    center_shadow_opacity: int
    center_highlight_strength: int
