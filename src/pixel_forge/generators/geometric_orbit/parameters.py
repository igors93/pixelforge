"""Typed domain models for the geometric-orbit generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Color = tuple[int, int, int]
ShapeKind = Literal["circle", "square", "triangle", "capsule", "line"]
CompositionStyle = Literal[
    "balanced-orbit",
    "capsule-sun",
    "bubble-ring",
    "shard-burst",
    "double-ring",
    "minimal-symbols",
]
BackgroundStyle = Literal[
    "diagonal-panels",
    "angular-folds",
    "offset-ribbons",
    "split-field",
]


@dataclass(frozen=True, slots=True)
class PaletteDefinition:
    """Complete color system used by one composition."""

    name: str
    canvas: Color
    panel_primary: Color
    panel_secondary: Color
    panel_accent: Color
    center_fill: Color
    center_outline: Color
    center_highlight: Color
    orbit_fill: Color
    orbit_outline: Color
    inner_fill: Color
    inner_outline: Color
    shadow: Color


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """One large geometric panel in normalized coordinates."""

    center_x: float
    center_y: float
    width: float
    height: float
    rotation_degrees: float
    color: Color
    corner_radius: float
    shadow_opacity: int


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


@dataclass(frozen=True, slots=True)
class GeometricOrbitParameters:
    """Fully sampled recipe used by the renderer without further randomness."""

    palette: PaletteDefinition
    composition_style: CompositionStyle
    background_style: BackgroundStyle
    panels: tuple[PanelSpec, ...]
    outer_shapes: tuple[ShapeSpec, ...]
    inner_shapes: tuple[ShapeSpec, ...]
    center_radius: float
    center_outline_width: float
    center_shadow_opacity: int
    center_highlight_strength: int
