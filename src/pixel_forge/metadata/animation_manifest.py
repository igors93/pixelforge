"""Animation manifest model and JSON serialization.

The manifest records the complete creative record of one generated GIF:
animation recipe, temporal quality, rarity, compatibility rules, and a
content identifier derived from the serialized recipes and all raw frame
bytes. The content identifier is updated incrementally as frames are
rendered to avoid keeping all raw bytes in memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixel_forge import __version__
from pixel_forge.core.models.animation_recipe import ANIMATION_SCHEMA_VERSION, AnimationRecipe
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION, ArtworkRecipe
from pixel_forge.core.models.rarity_result import RarityResult
from pixel_forge.core.models.temporal_quality_result import TemporalQualityResult
from pixel_forge.generators.common.types import UInt8Array


@dataclass(frozen=True, slots=True)
class AnimationManifest:
    """Complete record of a generated GIF suitable for JSON serialization."""

    pixelforge_version: str
    recipe_schema_version: str
    animation_schema_version: str
    generator: str
    width: int
    height: int
    frame_count: int
    fps: int
    frame_duration_ms: int
    total_duration_ms: int
    loop_count: int
    master_seed: int
    candidate_seed: int
    animation_seed: int
    motion_profile: str
    motion_intensity: float
    rotation_turns: int
    color_cycles: int
    pulse_count: int
    direction: str
    animated_traits: tuple[str, ...]
    static_rare_events: tuple[str, ...]
    animation_rare_events: tuple[str, ...]
    applied_compatibility_rules: tuple[str, ...]
    static_rarity_tier: str
    static_rarity_bits: float
    animation_rarity_tier: str
    animation_rarity_bits: float
    temporal_quality_score: float
    temporal_quality_accepted: bool
    temporal_quality_rejection_reasons: tuple[str, ...]
    gif_filename: str
    gif_colors: int
    gif_dither: str
    generated_at_utc: str
    content_id: str
    recipe_traits: dict[str, Any]
    animation_recipe_params: dict[str, Any]
    temporal_quality_metrics: dict[str, float]
    animation_rarity_details: dict[str, Any]


def build_animation_manifest(
    *,
    animation_recipe: AnimationRecipe,
    static_rarity: RarityResult,
    animation_rarity: RarityResult,
    temporal_quality: TemporalQualityResult,
    applied_rules: tuple[str, ...],
    gif_path: Path,
    gif_colors: int,
    gif_dither: str,
    content_id: str,
) -> AnimationManifest:
    """Construct an AnimationManifest from animation pipeline outputs."""
    timestamp = datetime.now(tz=UTC).isoformat()
    base = animation_recipe.base_recipe

    recipe_traits = _recipe_to_dict(base)
    anim_params = dict(animation_recipe.generator_animation_params)

    anim_rarity_details = {
        name: {
            "value": entry.value,
            "probability": entry.probability,
            "information_bits": entry.information_bits,
            "tier": _tier_for_prob(entry.probability),
        }
        for name, entry in animation_rarity.trait_details.items()
    }

    return AnimationManifest(
        pixelforge_version=__version__,
        recipe_schema_version=RECIPE_SCHEMA_VERSION,
        animation_schema_version=ANIMATION_SCHEMA_VERSION,
        generator=base.generator_name,
        width=base.width,
        height=base.height,
        frame_count=animation_recipe.frame_count,
        fps=animation_recipe.fps,
        frame_duration_ms=animation_recipe.frame_duration_ms,
        total_duration_ms=animation_recipe.frame_duration_ms * animation_recipe.frame_count,
        loop_count=animation_recipe.loop_count,
        master_seed=animation_recipe.master_seed,
        candidate_seed=base.candidate_seed,
        animation_seed=animation_recipe.animation_seed,
        motion_profile=animation_recipe.motion_profile,
        motion_intensity=animation_recipe.motion_intensity,
        rotation_turns=animation_recipe.rotation_turns,
        color_cycles=animation_recipe.color_cycles,
        pulse_count=animation_recipe.pulse_count,
        direction=animation_recipe.direction,
        animated_traits=animation_recipe.animated_traits,
        static_rare_events=base.rare_events,
        animation_rare_events=animation_recipe.rare_animation_events,
        applied_compatibility_rules=applied_rules,
        static_rarity_tier=static_rarity.overall_tier,
        static_rarity_bits=static_rarity.total_information_bits,
        animation_rarity_tier=animation_rarity.overall_tier,
        animation_rarity_bits=animation_rarity.total_information_bits,
        temporal_quality_score=temporal_quality.aggregate_score,
        temporal_quality_accepted=temporal_quality.accepted,
        temporal_quality_rejection_reasons=temporal_quality.rejection_reasons,
        gif_filename=gif_path.name,
        gif_colors=gif_colors,
        gif_dither=gif_dither,
        generated_at_utc=timestamp,
        content_id=content_id,
        recipe_traits=recipe_traits,
        animation_recipe_params=anim_params,
        temporal_quality_metrics=dict(temporal_quality.metrics),
        animation_rarity_details=anim_rarity_details,
    )


def compute_animation_content_id(
    animation_recipe: AnimationRecipe,
    frames: list[UInt8Array],
) -> str:
    """Compute a stable SHA-256 content ID from the recipe and all frame bytes.

    Updated incrementally to avoid holding all frames in memory simultaneously.
    """
    hasher = hashlib.sha256()
    # Stable recipe JSON
    recipe_dict = {
        "base_recipe": _recipe_to_dict(animation_recipe.base_recipe),
        "animation_schema_version": animation_recipe.animation_schema_version,
        "motion_profile": animation_recipe.motion_profile,
        "frame_count": animation_recipe.frame_count,
        "fps": animation_recipe.fps,
        "animation_seed": animation_recipe.animation_seed,
        "generator_animation_params": dict(animation_recipe.generator_animation_params),
    }
    hasher.update(json.dumps(recipe_dict, sort_keys=True, ensure_ascii=False).encode())
    for frame in frames:
        hasher.update(frame.tobytes())
    return hasher.hexdigest()


def manifest_to_json(manifest: AnimationManifest) -> str:
    """Serialize an AnimationManifest to stable JSON with sorted keys."""
    data: dict[str, Any] = {
        "pixelforge_version": manifest.pixelforge_version,
        "recipe_schema_version": manifest.recipe_schema_version,
        "animation_schema_version": manifest.animation_schema_version,
        "generator": manifest.generator,
        "width": manifest.width,
        "height": manifest.height,
        "frame_count": manifest.frame_count,
        "fps": manifest.fps,
        "frame_duration_ms": manifest.frame_duration_ms,
        "total_duration_ms": manifest.total_duration_ms,
        "loop_count": manifest.loop_count,
        "master_seed": manifest.master_seed,
        "candidate_seed": manifest.candidate_seed,
        "animation_seed": manifest.animation_seed,
        "motion_profile": manifest.motion_profile,
        "motion_intensity": manifest.motion_intensity,
        "rotation_turns": manifest.rotation_turns,
        "color_cycles": manifest.color_cycles,
        "pulse_count": manifest.pulse_count,
        "direction": manifest.direction,
        "animated_traits": list(manifest.animated_traits),
        "static_rare_events": list(manifest.static_rare_events),
        "animation_rare_events": list(manifest.animation_rare_events),
        "applied_compatibility_rules": list(manifest.applied_compatibility_rules),
        "static_rarity": {
            "tier": manifest.static_rarity_tier,
            "information_bits": manifest.static_rarity_bits,
        },
        "animation_rarity": {
            "tier": manifest.animation_rarity_tier,
            "information_bits": manifest.animation_rarity_bits,
            "trait_details": manifest.animation_rarity_details,
        },
        "temporal_quality": {
            "score": manifest.temporal_quality_score,
            "accepted": manifest.temporal_quality_accepted,
            "rejection_reasons": list(manifest.temporal_quality_rejection_reasons),
            "metrics": manifest.temporal_quality_metrics,
        },
        "recipe_traits": manifest.recipe_traits,
        "animation_recipe_params": manifest.animation_recipe_params,
        "gif_filename": manifest.gif_filename,
        "gif_colors": manifest.gif_colors,
        "gif_dither": manifest.gif_dither,
        "generated_at_utc": manifest.generated_at_utc,
        "content_id": manifest.content_id,
    }
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _recipe_to_dict(recipe: ArtworkRecipe) -> dict[str, Any]:
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


def _tier_for_prob(probability: float) -> str:
    from pixel_forge.rarity.rarity_tier import tier_for_probability
    return tier_for_probability(probability).value
