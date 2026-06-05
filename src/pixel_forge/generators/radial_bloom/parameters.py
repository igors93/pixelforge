"""Typed, immutable parameter block for the Radial Bloom generator.

Using a typed dataclass eliminates string-key dictionary access in the renderer,
surfaces missing parameters as AttributeError at parse time, and gives Mypy full
coverage of every numerical parameter. The flat dict stored in ArtworkRecipe
(for JSON serialization) is derived from and parsed back into this type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RadialBloomParams:
    """All numerical and discrete parameters for one Radial Bloom render."""

    primary_petals: int
    secondary_petals: int
    crown_count: int
    radial_ripple_count: int
    petal_sharpness: float       # controls petal tip pointiness (replaces curvature)
    petal_radial_scale: float    # scales glow radius per petal (replaces petal_width)
    center_mode: str             # "glow" | "void" | "bright" | "dark-star"
    phyllotaxis: bool
    spiral_clockwise: bool
    ripple_frequency: float
    phase: float
    glow_spread: float
    hue_base: float              # hue origin for HSV coloring

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RadialBloomParams:
        """Parse a recipe generator_params dict into a typed instance."""
        return cls(
            primary_petals=int(d["primary_petals"]),
            secondary_petals=int(d["secondary_petals"]),
            crown_count=int(d["crown_count"]),
            radial_ripple_count=int(d["radial_ripple_count"]),
            petal_sharpness=float(d.get("petal_sharpness", d.get("petal_curvature", 1.0))),
            petal_radial_scale=float(d.get("petal_radial_scale", d.get("petal_width", 1.0))),
            center_mode=str(d["center_mode"]),
            phyllotaxis=bool(d["phyllotaxis"]),
            spiral_clockwise=bool(d["spiral_clockwise"]),
            ripple_frequency=float(d["ripple_frequency"]),
            phase=float(d["phase"]),
            glow_spread=float(d["glow_spread"]),
            hue_base=float(d["hue_base"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat JSON-compatible dict for ArtworkRecipe storage."""
        return {
            "primary_petals": self.primary_petals,
            "secondary_petals": self.secondary_petals,
            "crown_count": self.crown_count,
            "radial_ripple_count": self.radial_ripple_count,
            "petal_sharpness": self.petal_sharpness,
            "petal_radial_scale": self.petal_radial_scale,
            "center_mode": self.center_mode,
            "phyllotaxis": self.phyllotaxis,
            "spiral_clockwise": self.spiral_clockwise,
            "ripple_frequency": self.ripple_frequency,
            "phase": self.phase,
            "glow_spread": self.glow_spread,
            "hue_base": self.hue_base,
        }
