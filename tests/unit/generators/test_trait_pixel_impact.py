"""Tests proving that changing one recipe trait while holding all others constant
produces a measurable pixel difference.

Each test renders a base recipe, then renders with one trait changed, and asserts
that the resulting RGB arrays are not identical.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
from pixel_forge.generators.radial_bloom import RadialBloomGenerator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams


def _request(seed: int = 77, width: int = 32, height: int = 32) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name="test",
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )


def _build(generator_name: str, seed: int = 77) -> tuple[object, object, int]:
    """Return (generator, base_recipe, candidate_seed) for the given generator."""
    generators = {
        "harmonic-waves": HarmonicWavesGenerator(),
        "plasma-flow": PlasmaFlowGenerator(),
        "radial-bloom": RadialBloomGenerator(),
        "mandelbrot-dream": MandelbrotDreamGenerator(),
    }
    gen = generators[generator_name]
    candidate_seed = derive_candidate_seed(
        master_seed=seed,
        generator_name=generator_name,
        retry_index=0,
        schema_version="1.0",
    )
    streams = RandomStreams.from_seed(candidate_seed)
    recipe, _ = gen.build_recipe(  # type: ignore[union-attr]
        _request(seed), streams, candidate_seed=candidate_seed, retry_index=0
    )
    return gen, recipe, candidate_seed


def _pixels_differ(gen: object, r1: object, r2: object) -> bool:
    a = gen.render_recipe(r1)  # type: ignore[union-attr]
    b = gen.render_recipe(r2)  # type: ignore[union-attr]
    return not np.array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry mode
# ─────────────────────────────────────────────────────────────────────────────

def test_symmetry_mirror_h_changes_pixels() -> None:
    gen, base, _ = _build("harmonic-waves")
    if base.symmetry_mode == SymmetryMode.MIRROR_H:
        pytest.skip("base already uses MIRROR_H")
    modified = replace(base, symmetry_mode=SymmetryMode.MIRROR_H)  # type: ignore[call-overload]
    assert _pixels_differ(gen, base, modified)


def test_symmetry_mirror_v_changes_pixels() -> None:
    gen, base, _ = _build("harmonic-waves")
    if base.symmetry_mode == SymmetryMode.MIRROR_V:
        pytest.skip("base already uses MIRROR_V")
    modified = replace(base, symmetry_mode=SymmetryMode.MIRROR_V)  # type: ignore[call-overload]
    assert _pixels_differ(gen, base, modified)


# ─────────────────────────────────────────────────────────────────────────────
# Complexity level
# ─────────────────────────────────────────────────────────────────────────────

def test_complexity_minimal_vs_intricate_harmonic() -> None:
    gen, base, _ = _build("harmonic-waves")
    minimal = replace(base, complexity_level=ComplexityLevel.MINIMAL)  # type: ignore[call-overload]
    intricate = replace(base, complexity_level=ComplexityLevel.INTRICATE)  # type: ignore[call-overload]
    assert _pixels_differ(gen, minimal, intricate)


def test_complexity_minimal_vs_intricate_plasma() -> None:
    gen, base, _ = _build("plasma-flow")
    minimal = replace(base, complexity_level=ComplexityLevel.MINIMAL)  # type: ignore[call-overload]
    intricate = replace(base, complexity_level=ComplexityLevel.INTRICATE)  # type: ignore[call-overload]
    assert _pixels_differ(gen, minimal, intricate)


# ─────────────────────────────────────────────────────────────────────────────
# Lighting mode
# ─────────────────────────────────────────────────────────────────────────────

def test_lighting_flat_vs_radial_harmonic() -> None:
    gen, base, _ = _build("harmonic-waves")
    flat = replace(base, lighting_mode=LightingMode.FLAT)  # type: ignore[call-overload]
    radial = replace(base, lighting_mode=LightingMode.RADIAL)  # type: ignore[call-overload]
    assert _pixels_differ(gen, flat, radial)


def test_lighting_flat_vs_directional_plasma() -> None:
    gen, base, _ = _build("plasma-flow")
    flat = replace(base, lighting_mode=LightingMode.FLAT)  # type: ignore[call-overload]
    directional = replace(base, lighting_mode=LightingMode.DIRECTIONAL)  # type: ignore[call-overload]
    assert _pixels_differ(gen, flat, directional)


def test_lighting_flat_vs_ambient_radial_bloom() -> None:
    gen, base, _ = _build("radial-bloom")
    flat = replace(base, lighting_mode=LightingMode.FLAT)  # type: ignore[call-overload]
    ambient = replace(base, lighting_mode=LightingMode.AMBIENT)  # type: ignore[call-overload]
    assert _pixels_differ(gen, flat, ambient)


# ─────────────────────────────────────────────────────────────────────────────
# Background mode
# ─────────────────────────────────────────────────────────────────────────────

def test_background_dark_vs_void_harmonic() -> None:
    gen, base, _ = _build("harmonic-waves")
    dark = replace(base, background_mode=BackgroundMode.DARK)  # type: ignore[call-overload]
    void = replace(base, background_mode=BackgroundMode.VOID)  # type: ignore[call-overload]
    assert _pixels_differ(gen, dark, void)


def test_background_dark_vs_light_plasma() -> None:
    gen, base, _ = _build("plasma-flow")
    dark = replace(base, background_mode=BackgroundMode.DARK)  # type: ignore[call-overload]
    light = replace(base, background_mode=BackgroundMode.LIGHT)  # type: ignore[call-overload]
    assert _pixels_differ(gen, dark, light)


def test_background_dark_vs_gradient_radial_bloom() -> None:
    gen, base, _ = _build("radial-bloom")
    dark = replace(base, background_mode=BackgroundMode.DARK)  # type: ignore[call-overload]
    gradient = replace(base, background_mode=BackgroundMode.GRADIENT)  # type: ignore[call-overload]
    assert _pixels_differ(gen, dark, gradient)


# ─────────────────────────────────────────────────────────────────────────────
# Accent mode
# ─────────────────────────────────────────────────────────────────────────────

def test_accent_none_vs_highlights_harmonic() -> None:
    gen, base, _ = _build("harmonic-waves")
    none = replace(base, accent_mode=AccentMode.NONE)  # type: ignore[call-overload]
    highlights = replace(base, accent_mode=AccentMode.HIGHLIGHTS)  # type: ignore[call-overload]
    assert _pixels_differ(gen, none, highlights)


def test_accent_none_vs_sparks_plasma() -> None:
    gen, base, _ = _build("plasma-flow")
    none = replace(base, accent_mode=AccentMode.NONE)  # type: ignore[call-overload]
    sparks = replace(base, accent_mode=AccentMode.SPARKS)  # type: ignore[call-overload]
    assert _pixels_differ(gen, none, sparks)


def test_accent_none_vs_luminous_radial_bloom() -> None:
    gen, base, _ = _build("radial-bloom")
    none = replace(base, accent_mode=AccentMode.NONE)  # type: ignore[call-overload]
    luminous = replace(base, accent_mode=AccentMode.LUMINOUS)  # type: ignore[call-overload]
    assert _pixels_differ(gen, none, luminous)


# ─────────────────────────────────────────────────────────────────────────────
# Detail level (mandelbrot)
# ─────────────────────────────────────────────────────────────────────────────

def test_detail_low_vs_high_mandelbrot() -> None:
    gen, base, _ = _build("mandelbrot-dream")
    low = replace(base, detail_level=DetailLevel.LOW)  # type: ignore[call-overload]
    high = replace(base, detail_level=DetailLevel.HIGH)  # type: ignore[call-overload]
    assert _pixels_differ(gen, low, high)


# ─────────────────────────────────────────────────────────────────────────────
# Palette name
# ─────────────────────────────────────────────────────────────────────────────

def test_different_palettes_change_pixels_harmonic() -> None:
    gen, base, _ = _build("harmonic-waves")
    r1 = replace(base, palette_name="solar-flare")  # type: ignore[call-overload]
    r2 = replace(base, palette_name="ocean-depth")  # type: ignore[call-overload]
    assert _pixels_differ(gen, r1, r2)


def test_different_palettes_change_pixels_plasma() -> None:
    gen, base, _ = _build("plasma-flow")
    r1 = replace(base, palette_name="solar-flare")  # type: ignore[call-overload]
    r2 = replace(base, palette_name="forest-mist")  # type: ignore[call-overload]
    assert _pixels_differ(gen, r1, r2)
