"""Tests for artwork manifest construction and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.artwork_traits import (
    AccentMode,
    BackgroundMode,
    ComplexityLevel,
    DetailLevel,
    LightingMode,
    SymmetryMode,
)
from pixel_forge.core.models.quality_result import QualityResult
from pixel_forge.core.models.rarity_result import RarityResult
from pixel_forge.metadata.artwork_manifest import build_manifest, manifest_to_json


def _recipe() -> ArtworkRecipe:
    return ArtworkRecipe(
        schema_version=RECIPE_SCHEMA_VERSION,
        generator_name="radial-bloom",
        seed=42,
        candidate_seed=12345,
        retry_index=0,
        width=64,
        height=64,
        palette_name="ocean-depth",
        symmetry_mode=SymmetryMode.RADIAL,
        complexity_level=ComplexityLevel.MODERATE,
        detail_level=DetailLevel.MEDIUM,
        background_mode=BackgroundMode.DARK,
        lighting_mode=LightingMode.FLAT,
        accent_mode=AccentMode.NONE,
        rare_events=("orbital-halo",),
        generator_params={"primary_petals": 8, "phase": 1.23},
    )


def _quality() -> QualityResult:
    return QualityResult(
        luminance_contrast=0.5,
        clipped_black_ratio=0.1,
        clipped_white_ratio=0.05,
        mean_saturation=0.6,
        saturation_spread=0.2,
        color_diversity=0.7,
        visual_entropy=0.8,
        edge_density=0.2,
        center_border_balance=0.5,
        horizontal_symmetry=0.9,
        vertical_symmetry=0.85,
        aggregate_score=0.65,
        accepted=True,
        rejection_reasons=(),
        metrics={"luminance_contrast": 0.5},
    )


def _rarity() -> RarityResult:
    return RarityResult(
        overall_tier="Uncommon",
        total_information_bits=5.5,
        most_significant_traits=(),
        trait_details={},
        summary="Test summary.",
    )


def test_build_manifest_fields() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    manifest = build_manifest(
        recipe=_recipe(),
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=("some-rule",),
        png_path=Path("output/test.png"),
        rgb_array=rgb,
    )
    assert manifest.generator == "radial-bloom"
    assert manifest.master_seed == 42
    assert manifest.candidate_seed == 12345
    assert manifest.retry_index == 0
    assert manifest.palette == "ocean-depth"
    assert "orbital-halo" in manifest.rare_events
    assert "some-rule" in manifest.applied_compatibility_rules
    assert manifest.quality_accepted
    assert manifest.rarity_tier == "Uncommon"


def test_manifest_json_is_valid() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    manifest = build_manifest(
        recipe=_recipe(),
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb,
    )
    json_str = manifest_to_json(manifest)
    data = json.loads(json_str)
    assert data["generator"] == "radial-bloom"
    assert data["master_seed"] == 42


def test_manifest_json_has_sorted_keys() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    manifest = build_manifest(
        recipe=_recipe(),
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb,
    )
    json_str = manifest_to_json(manifest)
    data = json.loads(json_str)
    keys = list(data.keys())
    assert keys == sorted(keys)


def test_content_id_is_stable() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    recipe = _recipe()
    manifest1 = build_manifest(
        recipe=recipe,
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb,
    )
    manifest2 = build_manifest(
        recipe=recipe,
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb,
    )
    assert manifest1.content_id == manifest2.content_id


def test_content_id_changes_with_different_pixels() -> None:
    recipe = _recipe()
    rgb_a = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_b = np.ones((64, 64, 3), dtype=np.uint8) * 128

    m_a = build_manifest(
        recipe=recipe,
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb_a,
    )
    m_b = build_manifest(
        recipe=recipe,
        rarity=_rarity(),
        quality=_quality(),
        applied_rules=(),
        png_path=Path("output/test.png"),
        rgb_array=rgb_b,
    )
    assert m_a.content_id != m_b.content_id
