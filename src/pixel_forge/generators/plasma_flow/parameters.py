"""Typed, immutable parameter block for the Plasma Flow generator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VortexEntry:
    """One vortex center and its rotation parameters."""

    x: float
    y: float
    strength: float
    sign: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VortexEntry:
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            strength=float(d["strength"]),
            sign=float(d["sign"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "strength": self.strength, "sign": self.sign}


@dataclass(frozen=True, slots=True)
class PlasmaFlowParams:
    """All numerical and discrete parameters for one Plasma Flow render."""

    warp_stages: int
    vortex_count: int
    flow_direction: str          # "radial" | "diagonal" | "horizontal" | "turbulent"
    phases: tuple[float, ...]
    center_x: float
    center_y: float
    freq_low: float
    freq_high: float
    warp_strength: float
    turbulence: float
    vortex_data: tuple[VortexEntry, ...]
    palette_phase_shift: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PlasmaFlowParams:
        phases_raw = d["phases"]
        vortex_raw = d.get("vortex_data", [])
        return cls(
            warp_stages=int(d["warp_stages"]),
            vortex_count=int(d["vortex_count"]),
            flow_direction=str(d["flow_direction"]),
            phases=tuple(float(v) for v in phases_raw),
            center_x=float(d["center_x"]),
            center_y=float(d["center_y"]),
            freq_low=float(d["freq_low"]),
            freq_high=float(d["freq_high"]),
            warp_strength=float(d["warp_strength"]),
            turbulence=float(d["turbulence"]),
            vortex_data=tuple(VortexEntry.from_dict(v) for v in vortex_raw),
            palette_phase_shift=float(d["palette_phase_shift"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "warp_stages": self.warp_stages,
            "vortex_count": self.vortex_count,
            "flow_direction": self.flow_direction,
            "phases": list(self.phases),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "freq_low": self.freq_low,
            "freq_high": self.freq_high,
            "warp_strength": self.warp_strength,
            "turbulence": self.turbulence,
            "vortex_data": [v.to_dict() for v in self.vortex_data],
            "palette_phase_shift": self.palette_phase_shift,
        }
