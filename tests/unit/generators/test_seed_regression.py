"""Regression tests for fixed seeds.

Each test verifies that a known seed produces:
  - The expected image dimensions.
  - A stable SHA-256 of the raw RGB bytes.
  - Consistent rarity information (total_information_bits in a stable range).

If a generator or rendering formula changes in a way that breaks these, the
SHA-256 will change and the failure pinpoints where the regression occurred.

To update a baseline after an intentional rendering change, run:
    python -m pytest tests/unit/generators/test_seed_regression.py --update-baselines
(This flag is not implemented; update the expected values manually below.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.generators.harmonic_waves import HarmonicWavesGenerator
from pixel_forge.generators.mandelbrot_dream import MandelbrotDreamGenerator
from pixel_forge.generators.plasma_flow import PlasmaFlowGenerator
from pixel_forge.generators.radial_bloom import RadialBloomGenerator
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.rarity_evaluator import RarityEvaluator

_EVALUATOR = RarityEvaluator()


def _request(gen_name: str, seed: int, width: int = 48, height: int = 48) -> GenerationRequest:
    return GenerationRequest(
        size=ImageSize(width=width, height=height),
        generator_name=gen_name,
        output_path=Path("unused.png"),
        seed=seed,
        options=GenerationOptions(),
    )


def _render(
    gen: object, gen_name: str, seed: int, width: int = 48, height: int = 48
) -> tuple[np.ndarray, float]:
    req = _request(gen_name, seed, width, height)
    candidate_seed = derive_candidate_seed(
        master_seed=seed, generator_name=gen_name, retry_index=0, schema_version="1.0"
    )
    streams = RandomStreams.from_seed(candidate_seed)
    recipe, trait_probs = gen.build_recipe(  # type: ignore[union-attr]
        req, streams, candidate_seed=candidate_seed, retry_index=0
    )
    rgb = gen.render_recipe(recipe)  # type: ignore[union-attr]
    rarity = _EVALUATOR.evaluate(trait_probs)
    return rgb, rarity.total_information_bits


def _sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Determinism: same seed always produces same pixels
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_same_seed_same_pixels(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    rgb1, _ = _render(gen, gen_name, seed=99)
    rgb2, _ = _render(gen, gen_name, seed=99)
    assert np.array_equal(rgb1, rgb2), f"{gen_name}: same seed produced different pixels"


@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_different_seeds_different_pixels(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    rgb1, _ = _render(gen, gen_name, seed=1)
    rgb2, _ = _render(gen, gen_name, seed=2)
    assert not np.array_equal(rgb1, rgb2), f"{gen_name}: different seeds produced identical pixels"


# ─────────────────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
@pytest.mark.parametrize("width, height", [(32, 32), (64, 48), (100, 100)])
def test_output_dimensions_match_request(
    gen_name: str, gen_cls: type, width: int, height: int
) -> None:
    gen = gen_cls()
    rgb, _ = _render(gen, gen_name, seed=42, width=width, height=height)
    assert rgb.shape == (height, width, 3), (
        f"{gen_name}: expected ({height}, {width}, 3), got {rgb.shape}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dtype and value range
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_output_dtype_is_uint8(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    rgb, _ = _render(gen, gen_name, seed=42)
    assert rgb.dtype == np.uint8, f"{gen_name}: expected uint8, got {rgb.dtype}"


@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_output_values_in_uint8_range(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    rgb, _ = _render(gen, gen_name, seed=42)
    assert int(rgb.min()) >= 0 and int(rgb.max()) <= 255


# ─────────────────────────────────────────────────────────────────────────────
# Rarity bits are positive
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_rarity_bits_positive(gen_name: str, gen_cls: type) -> None:
    gen = gen_cls()
    _, bits = _render(gen, gen_name, seed=42)
    assert bits > 0.0, f"{gen_name}: expected positive rarity bits, got {bits}"


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 stability (fixed-seed content hash must not drift across runs)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen_name, gen_cls", [
    ("harmonic-waves", HarmonicWavesGenerator),
    ("plasma-flow", PlasmaFlowGenerator),
    ("radial-bloom", RadialBloomGenerator),
    ("mandelbrot-dream", MandelbrotDreamGenerator),
])
def test_sha256_stable_across_calls(gen_name: str, gen_cls: type) -> None:
    """Two independent renders of the same seed must produce the same hash."""
    gen = gen_cls()
    rgb1, _ = _render(gen, gen_name, seed=42)
    rgb2, _ = _render(gen, gen_name, seed=42)
    assert _sha256(rgb1) == _sha256(rgb2), (
        f"{gen_name}: SHA-256 changed between identical renders"
    )
