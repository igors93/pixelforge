"""Application service that coordinates image generation.

New pipeline (recipe-driven generators):
  1. Validate request.
  2. Resolve generator.
  3. Build independent random streams from the effective seed.
  4. Build recipe (all random decisions, no pixels yet).
  5. Apply compatibility rules (deterministic modifications).
  6. Render RGB array from the recipe (no RNG calls).
  7. Evaluate quality heuristics.
  8. Retry deterministically if quality is below threshold.
  9. Encode PNG.
 10. Write PNG atomically.
 11. Write JSON manifest atomically (unless --no-metadata).
 12. Return expanded GenerationResult.

Legacy generators that only implement ImageGenerator (not RecipeGenerator)
continue to use the simpler single-call path for backward compatibility.
"""

from __future__ import annotations

import secrets

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.aesthetics.quality.quality_evaluator import QualityEvaluator, QualityThresholds
from pixel_forge.core.exceptions import QualityRejectionError
from pixel_forge.core.models import GeneratedImage, GenerationRequest, GenerationResult
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
from pixel_forge.shared.paths import normalize_output_path
from pixel_forge.shared.validation import RequestValidator


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
        self._validator.validate(request)
        output_path = normalize_output_path(
            request.output_path,
            supported_suffixes=self._supported_output_suffixes,
        )

        generator = self._registry.get(request.generator_name)
        master_seed = request.seed if request.seed is not None else secrets.randbits(64)

        # Override quality threshold from options if provided.
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
        else:
            return self._generate_legacy_pipeline(
                request=request,
                generator=generator,
                output_path=output_path,
            )

    # ------------------------------------------------------------------ #
    # Recipe-driven pipeline                                               #
    # ------------------------------------------------------------------ #

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

        # In strict quality mode, fail if no candidate was accepted.
        if request.options.strict_quality and accepted_rgb is None:
            reasons = best_quality.rejection_reasons if best_quality else ()
            raise QualityRejectionError(
                f"All {max_retries + 1} candidates failed quality checks. "
                + " ".join(reasons)
            )

        # Fall back to best candidate if strict quality is not enabled.
        final_rgb = accepted_rgb if accepted_rgb is not None else best_rgb
        final_recipe = accepted_recipe if accepted_recipe is not None else best_recipe
        final_rarity = accepted_rarity if accepted_rarity is not None else best_rarity
        final_quality = accepted_quality if accepted_quality is not None else best_quality
        final_rules = accepted_rules if accepted_rgb is not None else best_applied_rules

        assert final_rgb is not None
        assert final_recipe is not None
        assert final_rarity is not None
        assert final_quality is not None

        # Build GeneratedImage for encoding.
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

    # ------------------------------------------------------------------ #
    # Legacy single-call pipeline                                          #
    # ------------------------------------------------------------------ #

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
