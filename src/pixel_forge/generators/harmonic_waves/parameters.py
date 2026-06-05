"""Typed, immutable parameter block for the Harmonic Waves generator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HarmonicWavesParams:
    """All numerical and discrete parameters for one Harmonic Waves render."""

    layer_count: int
    warp_stages: int
    freq_set_name: str       # "harmonic" | "golden" | "sqrt-roots" | "pi-series"
    rotation: float
    warp_strength: float
    warp_freq_x: float
    warp_freq_y: float
    freq_scale: float
    phases: tuple[float, ...]
    palette_phase_d: tuple[float, float, float]
    primary_frequency: float  # used by compatibility rules

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HarmonicWavesParams:
        phases_raw = d["phases"]
        pd_raw = d["palette_phase_d"]
        return cls(
            layer_count=int(d["layer_count"]),
            warp_stages=int(d["warp_stages"]),
            freq_set_name=str(d["freq_set_name"]),
            rotation=float(d["rotation"]),
            warp_strength=float(d["warp_strength"]),
            warp_freq_x=float(d["warp_freq_x"]),
            warp_freq_y=float(d["warp_freq_y"]),
            freq_scale=float(d["freq_scale"]),
            phases=tuple(float(v) for v in phases_raw),
            palette_phase_d=(float(pd_raw[0]), float(pd_raw[1]), float(pd_raw[2])),
            primary_frequency=float(d.get("primary_frequency", d["freq_scale"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_count": self.layer_count,
            "warp_stages": self.warp_stages,
            "freq_set_name": self.freq_set_name,
            "rotation": self.rotation,
            "warp_strength": self.warp_strength,
            "warp_freq_x": self.warp_freq_x,
            "warp_freq_y": self.warp_freq_y,
            "freq_scale": self.freq_scale,
            "phases": list(self.phases),
            "palette_phase_d": list(self.palette_phase_d),
            "primary_frequency": self.primary_frequency,
        }
