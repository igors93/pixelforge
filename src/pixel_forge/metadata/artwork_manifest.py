"""Artwork manifest model and JSON serialization.

The manifest captures the complete creative record of one generated image:
recipe, rarity, quality, compatibility, and a content identifier derived from
stable data. The generation timestamp is recorded for human reference but is
never used in image computation.

The content identifier is a SHA-256 hash of the serialized recipe and a sample
of the rendered pixel bytes, providing a stable fingerprint that does not
depend on the PNG encoder metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixel_forge import __version__
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.quality_result import QualityResult
from pixel_forge.core.models.rarity_result import RarityResult
from pixel_forge.generators.common.types import UInt8Array


@dataclass(frozen=True, slots=True)
class ArtworkManifest:
    """Complete record of a generated image suitable for JSON serialization."""

    pixelforge_version: str
    recipe_schema_version: str
    generator: str
    width: int
    height: int
    master_seed: int
    candidate_seed: int
    retry_index: int
    palette: str
    rare_events: tuple[str, ...]
    applied_compatibility_rules: tuple[str, ...]
    rarity_tier: str
    rarity_information_bits: float
    rarity_summary: str
    quality_score: float
    quality_accepted: bool
    quality_rejection_reasons: tuple[str, ...]
    png_filename: str
    generated_at_utc: str
    content_id: str
    # Detailed nested structures stored as plain dicts for JSON compatibility.
    recipe_traits: dict[str, Any]
    quality_metrics: dict[str, float]
    rarity_trait_details: dict[str, Any]


def build_manifest(
    *,
    recipe: ArtworkRecipe,
    rarity: RarityResult,
    quality: QualityResult,
    applied_rules: tuple[str, ...],
    png_path: Path,
    rgb_array: UInt8Array,
) -> ArtworkManifest:
    """Construct a manifest from the generation pipeline outputs."""
    timestamp = datetime.now(tz=UTC).isoformat()

    recipe_traits = _recipe_to_dict(recipe)
    content_id = _compute_content_id(recipe_traits, rgb_array)

    rarity_trait_details = {
        name: {
            "value": entry.value,
            "probability": entry.probability,
            "information_bits": entry.information_bits,
            "tier": _tier_for_bits(entry.probability),
        }
        for name, entry in rarity.trait_details.items()
    }

    return ArtworkManifest(
        pixelforge_version=__version__,
        recipe_schema_version=RECIPE_SCHEMA_VERSION,
        generator=recipe.generator_name,
        width=recipe.width,
        height=recipe.height,
        master_seed=recipe.seed,
        candidate_seed=recipe.candidate_seed,
        retry_index=recipe.retry_index,
        palette=recipe.palette_name,
        rare_events=recipe.rare_events,
        applied_compatibility_rules=applied_rules,
        rarity_tier=rarity.overall_tier,
        rarity_information_bits=rarity.total_information_bits,
        rarity_summary=rarity.summary,
        quality_score=quality.aggregate_score,
        quality_accepted=quality.accepted,
        quality_rejection_reasons=quality.rejection_reasons,
        png_filename=png_path.name,
        generated_at_utc=timestamp,
        content_id=content_id,
        recipe_traits=recipe_traits,
        quality_metrics=dict(quality.metrics),
        rarity_trait_details=rarity_trait_details,
    )


def manifest_to_json(manifest: ArtworkManifest) -> str:
    """Serialize a manifest to a stable JSON string with sorted keys."""
    data: dict[str, Any] = {
        "pixelforge_version": manifest.pixelforge_version,
        "recipe_schema_version": manifest.recipe_schema_version,
        "generator": manifest.generator,
        "width": manifest.width,
        "height": manifest.height,
        "master_seed": manifest.master_seed,
        "candidate_seed": manifest.candidate_seed,
        "retry_index": manifest.retry_index,
        "palette": manifest.palette,
        "rare_events": list(manifest.rare_events),
        "applied_compatibility_rules": list(manifest.applied_compatibility_rules),
        "rarity": {
            "tier": manifest.rarity_tier,
            "information_bits": manifest.rarity_information_bits,
            "summary": manifest.rarity_summary,
            "trait_details": manifest.rarity_trait_details,
        },
        "quality": {
            "score": manifest.quality_score,
            "accepted": manifest.quality_accepted,
            "rejection_reasons": list(manifest.quality_rejection_reasons),
            "metrics": manifest.quality_metrics,
        },
        "recipe_traits": manifest.recipe_traits,
        "png_filename": manifest.png_filename,
        "generated_at_utc": manifest.generated_at_utc,
        "content_id": manifest.content_id,
    }
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _recipe_to_dict(recipe: ArtworkRecipe) -> dict[str, Any]:
    """Convert a recipe to a JSON-serializable dict."""
    return {
        "schema_version": recipe.schema_version,
        "generator_name": recipe.generator_name,
        "seed": recipe.seed,
        "candidate_seed": recipe.candidate_seed,
        "retry_index": recipe.retry_index,
        "width": recipe.width,
        "height": recipe.height,
        "palette_name": recipe.palette_name,
        "symmetry_mode": recipe.symmetry_mode.value,
        "complexity_level": recipe.complexity_level.value,
        "detail_level": recipe.detail_level.value,
        "background_mode": recipe.background_mode.value,
        "lighting_mode": recipe.lighting_mode.value,
        "accent_mode": recipe.accent_mode.value,
        "rare_events": list(recipe.rare_events),
        "generator_params": dict(recipe.generator_params),
    }


def _compute_content_id(recipe_dict: dict[str, Any], rgb_array: UInt8Array) -> str:
    """Compute a SHA-256 content identifier from stable recipe data and pixel bytes."""
    stable_recipe = json.dumps(recipe_dict, sort_keys=True, ensure_ascii=False)
    # Sample every 64th pixel to keep hashing fast even on large images.
    pixel_sample = rgb_array.ravel()[::64].tobytes()
    digest = hashlib.sha256(stable_recipe.encode() + pixel_sample).hexdigest()
    return digest


def _tier_for_bits(probability: float) -> str:
    from pixel_forge.rarity.rarity_tier import tier_for_probability
    return tier_for_probability(probability).value
