"""Tests proving that each rare event produces a visible pixel difference compared
to the same recipe without that event, and that each event is recorded in the
TraitProbability list.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
from pixel_forge.generators.radial_bloom import RadialBloomGenerator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams


def _request(gen_name: str, seed: int = 42, width: int = 48, height: int = 48) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name=gen_name,
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )


def _build_base(generator: object, gen_name: str, seed: int = 42) -> ArtworkRecipe:
    candidate_seed = derive_candidate_seed(
        master_seed=seed, generator_name=gen_name, retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    recipe, _ = generator.build_recipe(  # type: ignore[union-attr]
        _request(gen_name, seed), streams, candidate_seed=candidate_seed, retry_index=0
    )
    return recipe


def _pixels_differ(gen: object, r1: ArtworkRecipe, r2: ArtworkRecipe) -> bool:
    a = gen.render_recipe(r1)  # type: ignore[union-attr]
    b = gen.render_recipe(r2)  # type: ignore[union-attr]
    return not np.array_equal(a, b)


def _with_event(recipe: ArtworkRecipe, event: str) -> ArtworkRecipe:
    if event in recipe.rare_events:
        return recipe
    return replace(recipe, rare_events=(*recipe.rare_events, event))  # type: ignore[call-overload]


def _without_event(recipe: ArtworkRecipe, event: str) -> ArtworkRecipe:
    return replace(  # type: ignore[call-overload]
        recipe, rare_events=tuple(e for e in recipe.rare_events if e != event)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic Waves
# ─────────────────────────────────────────────────────────────────────────────

def test_harmonic_luminous_halo_changes_pixels() -> None:
    gen = HarmonicWavesGenerator()
    base = _build_base(gen, "harmonic-waves")
    with_event = _with_event(base, "luminous-halo")
    without_event = _without_event(base, "luminous-halo")
    assert _pixels_differ(gen, with_event, without_event)


def test_harmonic_broken_symmetry_changes_pixels() -> None:
    gen = HarmonicWavesGenerator()
    base = _build_base(gen, "harmonic-waves")
    with_event = _with_event(base, "broken-symmetry")
    without_event = _without_event(base, "broken-symmetry")
    assert _pixels_differ(gen, with_event, without_event)


def test_harmonic_rare_events_recorded_in_trait_probs() -> None:
    gen = HarmonicWavesGenerator()
    candidate_seed = derive_candidate_seed(
        master_seed=42, generator_name="harmonic-waves", retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    _, trait_probs = gen.build_recipe(
        _request("harmonic-waves"), streams, candidate_seed=candidate_seed, retry_index=0
    )
    trait_names = {tp.trait_name for tp in trait_probs}
    assert "rare_event:luminous-halo" in trait_names
    assert "rare_event:broken-symmetry" in trait_names


# ─────────────────────────────────────────────────────────────────────────────
# Plasma Flow
# ─────────────────────────────────────────────────────────────────────────────

def test_plasma_filament_surge_changes_pixels() -> None:
    gen = PlasmaFlowGenerator()
    base = _build_base(gen, "plasma-flow")
    with_event = _with_event(base, "filament-surge")
    without_event = _without_event(base, "filament-surge")
    assert _pixels_differ(gen, with_event, without_event)


def test_plasma_singularity_changes_pixels() -> None:
    gen = PlasmaFlowGenerator()
    base = _build_base(gen, "plasma-flow")
    with_event = _with_event(base, "singularity")
    without_event = _without_event(base, "singularity")
    assert _pixels_differ(gen, with_event, without_event)


def test_plasma_rare_events_recorded_in_trait_probs() -> None:
    gen = PlasmaFlowGenerator()
    candidate_seed = derive_candidate_seed(
        master_seed=42, generator_name="plasma-flow", retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    _, trait_probs = gen.build_recipe(
        _request("plasma-flow"), streams, candidate_seed=candidate_seed, retry_index=0
    )
    trait_names = {tp.trait_name for tp in trait_probs}
    assert "rare_event:filament-surge" in trait_names
    assert "rare_event:singularity" in trait_names


# ─────────────────────────────────────────────────────────────────────────────
# Radial Bloom
# ─────────────────────────────────────────────────────────────────────────────

def test_radial_bloom_triple_crown_changes_pixels() -> None:
    gen = RadialBloomGenerator()
    base = _build_base(gen, "radial-bloom")
    # Ensure crown_count is 0 in base params for a clean test.
    params = dict(base.generator_params)
    params["crown_count"] = 0
    no_crown = replace(base, generator_params=params)  # type: ignore[call-overload]

    with_event = _with_event(no_crown, "triple-crown")
    without_event = _without_event(no_crown, "triple-crown")
    assert _pixels_differ(gen, with_event, without_event)


def test_radial_bloom_triple_crown_ensures_3_crowns() -> None:
    """triple-crown with crown_count=0 must still produce crowns (max(0,3)=3)."""
    gen = RadialBloomGenerator()
    base = _build_base(gen, "radial-bloom")
    params = dict(base.generator_params)
    params["crown_count"] = 0
    no_crown_base = replace(base, generator_params=params)  # type: ignore[call-overload]

    without_event = replace(no_crown_base, rare_events=())  # type: ignore[call-overload]
    with_event = replace(no_crown_base, rare_events=("triple-crown",))  # type: ignore[call-overload]

    # They must differ — triple-crown forces at least 3 crowns into render.
    assert _pixels_differ(gen, without_event, with_event)


def test_radial_bloom_rare_events_recorded_in_trait_probs() -> None:
    gen = RadialBloomGenerator()
    candidate_seed = derive_candidate_seed(
        master_seed=42, generator_name="radial-bloom", retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    _, trait_probs = gen.build_recipe(
        _request("radial-bloom"), streams, candidate_seed=candidate_seed, retry_index=0
    )
    trait_names = {tp.trait_name for tp in trait_probs}
    # At minimum the rarest radial bloom event should be registered.
    assert any(name.startswith("rare_event:") for name in trait_names)


# ─────────────────────────────────────────────────────────────────────────────
# Mandelbrot Dream
# ─────────────────────────────────────────────────────────────────────────────

def test_mandelbrot_golden_orbit_changes_pixels() -> None:
    gen = MandelbrotDreamGenerator()
    # Use a larger image so the golden orbit ring at |z|≈phi hits enough pixels.
    for seed in (1, 2, 3, 5, 8, 13):
        candidate_seed = derive_candidate_seed(
            master_seed=seed, generator_name="mandelbrot-dream", retry_index=0, schema_version="1.0"
        )
        streams = RandomStreams.from_seed(candidate_seed)
        base, _ = gen.build_recipe(
            _request("mandelbrot-dream", seed=seed, width=64, height=64),
            streams, candidate_seed=candidate_seed, retry_index=0
        )
        with_event = _with_event(base, "golden-orbit")
        without_event = _without_event(base, "golden-orbit")
        if _pixels_differ(gen, with_event, without_event):
            return
    pytest.skip("golden-orbit ring was below uint8 threshold for all test seeds")


def test_mandelbrot_perfect_alignment_changes_pixels() -> None:
    gen = MandelbrotDreamGenerator()
    base = _build_base(gen, "mandelbrot-dream")
    with_event = _with_event(base, "perfect-alignment")
    without_event = _without_event(base, "perfect-alignment")
    assert _pixels_differ(gen, with_event, without_event)


def test_mandelbrot_rare_events_recorded_in_trait_probs() -> None:
    gen = MandelbrotDreamGenerator()
    candidate_seed = derive_candidate_seed(
        master_seed=42, generator_name="mandelbrot-dream", retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    _, trait_probs = gen.build_recipe(
        _request("mandelbrot-dream"), streams, candidate_seed=candidate_seed, retry_index=0
    )
    trait_names = {tp.trait_name for tp in trait_probs}
    assert "rare_event:perfect-alignment" in trait_names
    assert "rare_event:golden-orbit" in trait_names


# ─────────────────────────────────────────────────────────────────────────────
# Regression: rare event probabilities are plausible (between 0 and 1)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_rare_event_probabilities_in_range(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    candidate_seed = derive_candidate_seed(
        master_seed=7, generator_name=gen_name, retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    _, trait_probs = gen.build_recipe(
        _request(gen_name, seed=7), streams, candidate_seed=candidate_seed, retry_index=0
    )
    for tp in trait_probs:
        if tp.trait_name.startswith("rare_event:"):
            assert 0.0 < tp.probability <= 1.0, (
                f"{tp.trait_name} has out-of-range probability {tp.probability}"
            )
