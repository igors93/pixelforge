"""Typed, immutable parameter block for the Mandelbrot Dream generator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MandelbrotDreamParams:
    """All numerical and discrete parameters for one Mandelbrot Dream render."""

    region: str           # named region of interest
    zoom: float
    center_real: float
    center_imag: float
    max_iterations: int
    interior_mode: str    # "black" | "nebula" | "white" | "dark-star"
    color_cycle: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MandelbrotDreamParams:
        return cls(
            region=str(d["region"]),
            zoom=float(d["zoom"]),
            center_real=float(d["center_real"]),
            center_imag=float(d["center_imag"]),
            max_iterations=int(d["max_iterations"]),
            interior_mode=str(d["interior_mode"]),
            color_cycle=float(d["color_cycle"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "zoom": self.zoom,
            "center_real": self.center_real,
            "center_imag": self.center_imag,
            "max_iterations": self.max_iterations,
            "interior_mode": self.interior_mode,
            "color_cycle": self.color_cycle,
        }
