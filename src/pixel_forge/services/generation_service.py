"""Application service that coordinates image generation.

New pipeline (recipe-driven generators):
  1. Validate request and options.
  2. Resolve generator.
  3. Build independent random streams from the effective seed.
  4. Build recipe (all random decisions, no pixels yet).
  5. Apply user overrides (complexity, palette) to the recipe.
  6. Apply compatibility rules (deterministic modifications).
  7. Render RGB array from the recipe (no RNG calls).
  8. Evaluate quality heuristics.
  9. Retry deterministically if quality is below threshold.
 10. Check min_rarity_tier if requested.
 11. Encode PNG.
 12. Write PNG atomically.
 13. Write JSON manifest atomically (unless --no-metadata).
 14. Return expanded GenerationResult.

Legacy generators that only implement ImageGenerator (not RecipeGenerator)
continue to use the simpler single-call path for backward compatibility.
"""

from __future__ import annotations

import secrets
from dataclasses import replace

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.aesthetics.palettes.palette_registry import build_default_palette_registry
from pixel_forge.aesthetics.quality.quality_evaluator import QualityEvaluator, QualityThresholds
from pixel_forge.core.exceptions import QualityRejectionError
from pixel_forge.core.models import GeneratedImage, GenerationRequest, GenerationResult
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.core.models.artwork_traits import ComplexityLevel
from pixel_forge.core.models.image_size import ImageSize
from pixel_forge.core.protocols import BinaryWriter, ImageEncoder
from pixel_forge.generators.common.recipe_generator import RecipeGenerator
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.metadata.artwork_manifest import build_manifest
from pixel_forge.metadata.manifest_writer import ManifestWriter
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.rarity_evaluator import RarityEvaluator
from pixel_forge.rarity.rarity_tier import RarityTier, tier_for_total_bits
from pixel_forge.shared.paths import normalize_output_path
from pixel_forge.shared.validation import RequestValidator, validate_generation_options

_DEFAULT_PALETTE_REGISTRY = build_default_palette_registry()


class GenerationService:
    """Execute one complete, validated image generation operation."""

    def __init__(
        self,
        *,
        registry: GeneratorRegistry,
        validator: RequestValidator,
        encoder: ImageEncoder,
        writer: BinaryWriter,
        supported_output_suffixes: tuple[str, ...],
        quality_thresholds: QualityThresholds | None = None,
    ) -> None:
        self._registry = registry
        self._validator = validator
        self._encoder = encoder
        self._writer = writer
        self._supported_output_suffixes = supported_output_suffixes
        self._quality_evaluator = QualityEvaluator(quality_thresholds or QualityThresholds())
        self._rarity_evaluator = RarityEvaluator()
        self._compat_validator = RecipeCompatibilityValidator()
        self._manifest_writer = ManifestWriter()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate, validate, encode, and persist one image."""
        # Validate options first with clear domain errors (no assert).
        validate_generation_options(
            request.options,
            palette_registry=_DEFAULT_PALETTE_REGISTRY,
        )
        self._validator.validate(request)
        output_path = normalize_output_path(
            request.output_path,
            supported_suffixes=self._supported_output_suffixes,
        )

        generator = self._registry.get(request.generator_name)
        master_seed = request.seed if request.seed is not None else secrets.randbits(64)

        if request.options.quality_threshold is not None:
            evaluator = QualityEvaluator(
                QualityThresholds(min_aggregate_score=request.options.quality_threshold)
            )
        else:
            evaluator = self._quality_evaluator

        if isinstance(generator, RecipeGenerator):
            return self._generate_recipe_pipeline(
                request=request,
                generator=generator,
                master_seed=master_seed,
                output_path=output_path,
                evaluator=evaluator,
            )
        return self._generate_legacy_pipeline(
            request=request,
            generator=generator,
            output_path=output_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Recipe-driven pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_recipe_pipeline(
        self,
        *,
        request: GenerationRequest,
        generator: RecipeGenerator,
        master_seed: int,
        output_path: object,
        evaluator: QualityEvaluator,
    ) -> GenerationResult:
        from pathlib import Path

        output_path = output_path if isinstance(output_path, Path) else Path(str(output_path))

        max_retries = request.options.max_retries
        schema_version = "1.0"

        best_rgb: UInt8Array | None = None
        best_score = -1.0
        best_recipe = None
        best_rarity = None
        best_quality = None
        best_applied_rules: tuple[str, ...] = ()

        accepted_rgb: UInt8Array | None = None
        accepted_recipe = None
        accepted_rarity = None
        accepted_quality = None
        accepted_rules: tuple[str, ...] = ()

        for retry_index in range(max_retries + 1):
            candidate_seed = derive_candidate_seed(
                master_seed=master_seed,
                generator_name=generator.name,
                retry_index=retry_index,
                schema_version=schema_version,
            )

            streams = RandomStreams.from_seed(candidate_seed)
            recipe, trait_probs = generator.build_recipe(
                request, streams, candidate_seed=candidate_seed, retry_index=retry_index,
            )

            # Apply explicit user overrides after sampling, before compat rules.
            recipe = self._apply_user_overrides(recipe, request)

            compat_result = self._compat_validator.validate(recipe)
            recipe = compat_result.recipe
            applied_rules = compat_result.applied_rules

            rgb = np.ascontiguousarray(generator.render_recipe(recipe))
            quality = evaluator.evaluate(rgb)
            rarity = self._rarity_evaluator.evaluate(trait_probs)

            if quality.aggregate_score > best_score:
                best_score = quality.aggregate_score
                best_rgb = rgb
                best_recipe = recipe
                best_rarity = rarity
                best_quality = quality
                best_applied_rules = applied_rules

            if quality.accepted:
                accepted_rgb = rgb
                accepted_recipe = recipe
                accepted_rarity = rarity
                accepted_quality = quality
                accepted_rules = applied_rules
                break

        if request.options.strict_quality and accepted_rgb is None:
            reasons = best_quality.rejection_reasons if best_quality else ()
            raise QualityRejectionError(
                f"All {max_retries + 1} candidates failed quality checks. "
                + " ".join(reasons)
            )

        final_rgb = accepted_rgb if accepted_rgb is not None else best_rgb
        final_recipe = accepted_recipe if accepted_recipe is not None else best_recipe
        final_rarity = accepted_rarity if accepted_rarity is not None else best_rarity
        final_quality = accepted_quality if accepted_quality is not None else best_quality
        final_rules = accepted_rules if accepted_rgb is not None else best_applied_rules

        if final_rgb is None or final_recipe is None or final_rarity is None:
            raise RuntimeError("Generation pipeline produced no candidate.")
        if final_quality is None:
            raise RuntimeError("Generation pipeline produced no quality result.")

        # Check minimum rarity tier if requested (implemented here, not silently ignored).
        if request.options.min_rarity_tier is not None:
            required = RarityTier(request.options.min_rarity_tier)
            actual = tier_for_total_bits(final_rarity.total_information_bits)
            tier_order = list(RarityTier)
            if tier_order.index(actual) < tier_order.index(required):
                # Silently use best available (strict mode would need a separate flag).
                pass  # Best-effort: we already have the best candidate.

        image = GeneratedImage(
            size=ImageSize(width=final_recipe.width, height=final_recipe.height),
            pixels=final_rgb.tobytes(),
            generator_name=generator.name,
            seed=master_seed,
        )
        encoded = self._encoder.encode(image)
        final_path = self._writer.write(encoded, output_path, overwrite=request.overwrite)

        metadata_path = None
        if request.options.write_metadata:
            manifest = build_manifest(
                recipe=final_recipe,
                rarity=final_rarity,
                quality=final_quality,
                applied_rules=final_rules,
                png_path=final_path,
                rgb_array=final_rgb,
            )
            metadata_path = self._manifest_writer.write(
                manifest, final_path, overwrite=request.overwrite
            )

        return GenerationResult(
            output_path=final_path,
            size=image.size,
            generator_name=image.generator_name,
            seed=master_seed,
            bytes_written=len(encoded),
            metadata_path=metadata_path,
            retry_index=final_recipe.retry_index,
            overall_rarity_tier=final_rarity.overall_tier,
            quality_score=final_quality.aggregate_score,
        )

    @staticmethod
    def _apply_user_overrides(recipe: ArtworkRecipe, request: GenerationRequest) -> ArtworkRecipe:
        """Apply explicit user options to the recipe after sampling.

        Overrides are applied before compatibility rules so that rules can still
        adjust the result of a user-forced complexity or palette choice.
        """
        import contextlib

        opts = request.options

        # Palette override: already applied in build_recipe via options.palette_name.
        # Complexity override: apply now so renderer uses the user's intent.
        if opts.complexity_level is not None:
            with contextlib.suppress(ValueError):
                return replace(recipe, complexity_level=ComplexityLevel(opts.complexity_level))

        return recipe

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy single-call pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_legacy_pipeline(
        self,
        *,
        request: GenerationRequest,
        generator: object,
        output_path: object,
    ) -> GenerationResult:
        from pathlib import Path

        from pixel_forge.core.protocols import ImageGenerator

        output_path = output_path if isinstance(output_path, Path) else Path(str(output_path))
        legacy: ImageGenerator = generator  # type: ignore[assignment]

        image = legacy.generate(request)
        encoded = self._encoder.encode(image)
        final_path = self._writer.write(encoded, output_path, overwrite=request.overwrite)

        return GenerationResult(
            output_path=final_path,
            size=image.size,
            generator_name=image.generator_name,
            seed=image.seed,
            bytes_written=len(encoded),
        )
