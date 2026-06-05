"""Application service that coordinates procedural GIF animation generation.

Pipeline:
  1. Validate the animation request and options.
  2. Resolve the static recipe generator.
  3. Resolve the generator-specific animated renderer.
  4. Build and validate the static artwork recipe.
  5. Build deterministic animation candidates.
  6. Render frames and evaluate temporal quality and rarity.
  7. Encode the accepted or best candidate as GIF.
  8. Persist the GIF and optional JSON manifest.

Static PNG output is not modified by this service.
"""

from __future__ import annotations

import os
import secrets
import tempfile
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.animation.animation_randomness import (
    AnimationStreams,
    derive_animation_retry_seed,
    derive_animation_seed,
)
from pixel_forge.animation.frame_phase import generate_frame_phases
from pixel_forge.animation.motion_profiles import ALL_PROFILES_BY_GENERATOR
from pixel_forge.animation.protocols import AnimatedRecipeRenderer
from pixel_forge.animation.quality import (
    TemporalQualityEvaluator,
    TemporalQualityThresholds,
)
from pixel_forge.core.config import Settings
from pixel_forge.core.exceptions import (
    OptionsValidationError,
    OutputFileExistsError,
    QualityRejectionError,
    UnsupportedOutputFormatError,
    ValidationError,
)
from pixel_forge.core.models.animation_recipe import (
    ANIMATION_SCHEMA_VERSION,
    AnimationRecipe,
)
from pixel_forge.core.models.animation_request import AnimationRequest
from pixel_forge.core.models.animation_result import AnimationResult
from pixel_forge.core.models.artwork_recipe import RECIPE_SCHEMA_VERSION
from pixel_forge.core.models.generation_options import GenerationOptions
from pixel_forge.core.models.generation_request import GenerationRequest
from pixel_forge.core.models.image_size import ImageSize
from pixel_forge.core.models.rarity_result import RarityResult
from pixel_forge.core.models.temporal_quality_result import TemporalQualityResult
from pixel_forge.generators.common.recipe_generator import RecipeGenerator
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.image.encoders.gif_encoder import GifEncoder
from pixel_forge.image.encoders.gif_encoding_options import GifEncodingOptions
from pixel_forge.image.writers.atomic_file_writer import AtomicFileWriter
from pixel_forge.metadata.animation_manifest import (
    build_animation_manifest,
    compute_animation_content_id,
)
from pixel_forge.metadata.animation_manifest_writer import AnimationManifestWriter
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.rarity_evaluator import RarityEvaluator
from pixel_forge.rarity.trait_probability import TraitProbability

# Animation-specific safety limits.
_MIN_FRAMES = 2
_MAX_FRAMES = 120
_MIN_FPS = 1
_MAX_FPS = 60
_MAX_ANIMATION_RETRIES = 20
_MAX_PIXEL_BUDGET = 512 * 512 * 120

_RARITY_RANK: dict[str, int] = {
    "Common": 0,
    "Uncommon": 1,
    "Rare": 2,
    "Epic": 3,
    "Legendary": 4,
}


def _build_renderer_registry() -> dict[str, AnimatedRecipeRenderer]:
    """Import and instantiate every available animated renderer."""
    from pixel_forge.generators.harmonic_waves.animation import HarmonicWavesAnimator
    from pixel_forge.generators.mandelbrot_dream.animation import MandelbrotDreamAnimator
    from pixel_forge.generators.plasma_flow.animation import PlasmaFlowAnimator
    from pixel_forge.generators.radial_bloom.animation import RadialBloomAnimator

    renderers: tuple[AnimatedRecipeRenderer, ...] = (
        HarmonicWavesAnimator(),
        RadialBloomAnimator(),
        PlasmaFlowAnimator(),
        MandelbrotDreamAnimator(),
    )
    return {renderer.name: renderer for renderer in renderers}


_RENDERER_REGISTRY = _build_renderer_registry()


class AnimationService:
    """Execute one complete and validated animation generation operation."""

    def __init__(
        self,
        *,
        registry: GeneratorRegistry,
        settings: Settings | None = None,
        writer: AtomicFileWriter | None = None,
        temporal_quality_thresholds: TemporalQualityThresholds | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings or Settings()
        self._writer = writer or AtomicFileWriter()
        self._compat_validator = RecipeCompatibilityValidator()
        self._rarity_evaluator = RarityEvaluator()
        self._temporal_evaluator = TemporalQualityEvaluator(
            temporal_quality_thresholds
        )
        self._manifest_writer = AnimationManifestWriter()

    def animate(self, request: AnimationRequest) -> AnimationResult:
        """Generate, validate, encode, and persist one animated GIF."""
        self._validate_request(request)

        output_path = request.output_path
        if output_path.suffix.lower() != ".gif":
            raise UnsupportedOutputFormatError(
                f"Animation output must be a .gif file, got: {output_path.suffix}"
            )

        metadata_output_path = output_path.with_suffix(".json")
        if output_path.exists() and not request.overwrite:
            raise OutputFileExistsError(
                f"Output file already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        if (
            request.options.write_metadata
            and metadata_output_path.exists()
            and not request.overwrite
        ):
            raise OutputFileExistsError(
                f"Metadata file already exists: {metadata_output_path}. "
                "Use --overwrite to replace it."
            )

        opts = request.options

        # Use the service default unless the user explicitly overrides the
        # aggregate temporal-quality threshold for this request.
        evaluator = self._temporal_evaluator
        if opts.temporal_quality_threshold is not None:
            evaluator = TemporalQualityEvaluator(
                TemporalQualityThresholds(
                    min_aggregate_score=opts.temporal_quality_threshold,
                )
            )

        generator = self._registry.get(request.generator_name)
        if not isinstance(generator, RecipeGenerator):
            raise ValidationError(
                f"Generator '{request.generator_name}' does not support animation "
                "because it is not recipe-driven."
            )

        renderer = _RENDERER_REGISTRY.get(request.generator_name)
        if renderer is None:
            supported = ", ".join(sorted(_RENDERER_REGISTRY))
            raise ValidationError(
                f"Generator '{request.generator_name}' does not have an animated "
                f"renderer. Supported animated generators: {supported}"
            )

        master_seed = (
            request.seed if request.seed is not None else secrets.randbits(64)
        )

        # Build the base static recipe once. Animation retries preserve this
        # artwork identity and only resample animation-specific decisions.
        candidate_seed = derive_candidate_seed(
            master_seed=master_seed,
            generator_name=generator.name,
            retry_index=0,
            schema_version=RECIPE_SCHEMA_VERSION,
        )
        static_streams = RandomStreams.from_seed(candidate_seed)
        static_recipe, static_trait_probs = generator.build_recipe(
            self._static_request(request),
            static_streams,
            candidate_seed=candidate_seed,
            retry_index=0,
        )

        compatibility = self._compat_validator.validate(static_recipe)
        static_recipe = compatibility.recipe
        applied_rules = compatibility.applied_rules
        static_rarity = self._rarity_evaluator.evaluate(static_trait_probs)

        max_retries = opts.max_animation_retries
        best_result: _CandidateResult | None = None
        best_score = -1.0
        accepted_result: _CandidateResult | None = None

        base_animation_seed = derive_animation_seed(
            candidate_seed=candidate_seed,
            generator_name=generator.name,
            animation_schema_version=ANIMATION_SCHEMA_VERSION,
        )

        for retry_index in range(max_retries + 1):
            animation_seed = base_animation_seed
            if retry_index > 0:
                animation_seed = derive_animation_retry_seed(
                    animation_seed=base_animation_seed,
                    retry_index=retry_index,
                )

            animation_streams = AnimationStreams.from_seed(animation_seed)
            animation_recipe, animation_trait_probs = (
                renderer.build_animation_recipe(
                    static_recipe,
                    opts,
                    animation_streams,
                    animation_seed,
                )
            )

            # Builders currently create retry_index=0. The service owns retry
            # orchestration, so it records the real attempt on the frozen recipe.
            animation_recipe = replace(
                animation_recipe,
                retry_index=retry_index,
            )

            animation_rarity = self._rarity_evaluator.evaluate(
                animation_trait_probs
            )
            frames = self._render_all_frames(renderer, animation_recipe)

            # Phase 1.0 is not encoded as a duplicate frame, but rendering it here
            # verifies that the mathematical path closes back onto phase 0.0.
            virtual_phase0 = np.ascontiguousarray(
                renderer.render_frame(animation_recipe, 0.0)
            )
            virtual_phase1 = np.ascontiguousarray(
                renderer.render_frame(animation_recipe, 1.0)
            )

            temporal_quality = evaluator.evaluate(
                frames,
                virtual_phase0=virtual_phase0,
                virtual_phase1=virtual_phase1,
            )

            current_result = _CandidateResult(
                animation_recipe=animation_recipe,
                animation_trait_probs=animation_trait_probs,
                animation_rarity=animation_rarity,
                frames=frames,
                temporal_quality=temporal_quality,
            )

            if temporal_quality.aggregate_score > best_score:
                best_score = temporal_quality.aggregate_score
                best_result = current_result

            rarity_accepted = self._rarity_is_accepted(
                actual_tier=animation_rarity.overall_tier,
                minimum_tier=opts.min_animation_rarity_tier,
            )
            if temporal_quality.accepted and rarity_accepted:
                accepted_result = current_result
                break

        if accepted_result is None and opts.min_animation_rarity_tier is not None:
            raise QualityRejectionError(
                "No animation candidate reached the requested minimum rarity "
                f"tier '{opts.min_animation_rarity_tier}' after "
                f"{max_retries + 1} attempts."
            )

        if opts.strict_temporal_quality and accepted_result is None:
            reasons = (
                best_result.temporal_quality.rejection_reasons
                if best_result is not None
                else ()
            )
            details = " ".join(reasons)
            raise QualityRejectionError(
                f"All {max_retries + 1} animation candidates failed temporal "
                f"quality. {details}".strip()
            )

        final = accepted_result if accepted_result is not None else best_result
        if final is None:
            raise RuntimeError("Animation service produced no candidate result.")

        gif_options = GifEncodingOptions(
            gif_colors=opts.gif_colors,
            dither=opts.gif_dither,
            loop_count=opts.loop_count,
        )
        encoder = GifEncoder(gif_options)
        gif_bytes = encoder.encode(
            final.frames,
            frame_duration_ms=final.animation_recipe.frame_duration_ms,
        )

        content_id = compute_animation_content_id(
            final.animation_recipe,
            final.frames,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path = self._write_gif(
            gif_bytes,
            output_path,
            overwrite=request.overwrite,
        )

        metadata_path: Path | None = None
        if opts.write_metadata:
            manifest = build_animation_manifest(
                animation_recipe=final.animation_recipe,
                static_rarity=static_rarity,
                animation_rarity=final.animation_rarity,
                temporal_quality=final.temporal_quality,
                applied_rules=applied_rules,
                gif_path=gif_path,
                gif_colors=opts.gif_colors,
                gif_dither=opts.gif_dither,
                content_id=content_id,
            )
            metadata_path = self._manifest_writer.write(
                manifest,
                gif_path,
                overwrite=request.overwrite,
            )

        return AnimationResult(
            output_path=gif_path,
            size=ImageSize(
                width=static_recipe.width,
                height=static_recipe.height,
            ),
            generator_name=generator.name,
            master_seed=master_seed,
            animation_seed=final.animation_recipe.animation_seed,
            frame_count=final.animation_recipe.frame_count,
            fps=final.animation_recipe.fps,
            frame_duration_ms=final.animation_recipe.frame_duration_ms,
            loop_count=final.animation_recipe.loop_count,
            motion_profile=final.animation_recipe.motion_profile,
            bytes_written=len(gif_bytes),
            metadata_path=metadata_path,
            retry_index=final.animation_recipe.retry_index,
            animation_rarity_tier=final.animation_rarity.overall_tier,
            temporal_quality_score=final.temporal_quality.aggregate_score,
            content_id=content_id,
        )

    def _render_all_frames(
        self,
        renderer: AnimatedRecipeRenderer,
        animation_recipe: AnimationRecipe,
    ) -> list[UInt8Array]:
        """Render every encoded phase without adding a duplicate endpoint."""
        phases = generate_frame_phases(animation_recipe.frame_count)
        return [
            np.ascontiguousarray(
                renderer.render_frame(animation_recipe, phase)
            )
            for phase in phases
        ]

    def _write_gif(
        self,
        gif_bytes: bytes,
        output_path: Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Write GIF bytes through a temporary file and return the final path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=output_path.parent,
            suffix=".gif.tmp",
        )

        try:
            with os.fdopen(file_descriptor, "wb") as output_file:
                output_file.write(gif_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())

            if output_path.exists() and overwrite:
                output_path.unlink()
            os.replace(temporary_path, output_path)
        except Exception:
            with suppress(OSError):
                os.unlink(temporary_path)
            raise

        return output_path

    def _validate_request(self, request: AnimationRequest) -> None:
        """Validate dimensions and animation-specific safety constraints."""
        options = request.options
        width = request.size.width
        height = request.size.height

        if not self._settings.min_width <= width <= self._settings.max_width:
            raise OptionsValidationError(
                f"width must be between {self._settings.min_width} and "
                f"{self._settings.max_width}, got {width}"
            )
        if not self._settings.min_height <= height <= self._settings.max_height:
            raise OptionsValidationError(
                f"height must be between {self._settings.min_height} and "
                f"{self._settings.max_height}, got {height}"
            )

        if options.frame_count < _MIN_FRAMES:
            raise OptionsValidationError(
                f"frame_count must be >= {_MIN_FRAMES}, got {options.frame_count}"
            )
        if options.frame_count > _MAX_FRAMES:
            raise OptionsValidationError(
                f"frame_count must be <= {_MAX_FRAMES}, got {options.frame_count}"
            )
        if options.fps < _MIN_FPS:
            raise OptionsValidationError(
                f"fps must be >= {_MIN_FPS}, got {options.fps}"
            )
        if options.fps > _MAX_FPS:
            raise OptionsValidationError(
                f"fps must be <= {_MAX_FPS}, got {options.fps}"
            )
        if options.gif_colors not in (64, 128, 256):
            raise OptionsValidationError(
                "gif_colors must be 64, 128, or 256, "
                f"got {options.gif_colors}"
            )
        if options.gif_dither not in ("none", "floyd-steinberg"):
            raise OptionsValidationError(
                "gif_dither must be 'none' or 'floyd-steinberg', "
                f"got {options.gif_dither!r}"
            )
        if options.loop_count < 0:
            raise OptionsValidationError(
                "loop_count must be >= 0 (0 means infinite), "
                f"got {options.loop_count}"
            )
        if options.max_animation_retries < 0:
            raise OptionsValidationError(
                "max_animation_retries must be >= 0, "
                f"got {options.max_animation_retries}"
            )
        if options.max_animation_retries > _MAX_ANIMATION_RETRIES:
            raise OptionsValidationError(
                "max_animation_retries must be <= "
                f"{_MAX_ANIMATION_RETRIES}, got "
                f"{options.max_animation_retries}"
            )
        if (
            options.motion_intensity is not None
            and not 0.0 <= options.motion_intensity <= 1.0
        ):
            raise OptionsValidationError(
                "motion_intensity must be between 0.0 and 1.0"
            )
        if (
            options.temporal_quality_threshold is not None
            and not 0.0 <= options.temporal_quality_threshold <= 1.0
        ):
            raise OptionsValidationError(
                "temporal_quality_threshold must be between 0.0 and 1.0"
            )
        if (
            options.min_animation_rarity_tier is not None
            and options.min_animation_rarity_tier not in _RARITY_RANK
        ):
            valid_tiers = ", ".join(_RARITY_RANK)
            raise OptionsValidationError(
                "min_animation_rarity_tier must be one of: "
                f"{valid_tiers}"
            )

        pixel_budget = width * height * options.frame_count
        if pixel_budget > _MAX_PIXEL_BUDGET:
            raise OptionsValidationError(
                f"Animation pixel budget {pixel_budget:,} exceeds maximum "
                f"{_MAX_PIXEL_BUDGET:,}. Reduce width, height, or frame_count."
            )

        if options.motion_profile is not None:
            valid_profiles = ALL_PROFILES_BY_GENERATOR.get(
                request.generator_name,
                (),
            )
            if options.motion_profile not in valid_profiles:
                valid_text = ", ".join(valid_profiles) or "(none)"
                raise OptionsValidationError(
                    f"motion_profile {options.motion_profile!r} is not valid for "
                    f"'{request.generator_name}'. Valid profiles: {valid_text}"
                )

        if request.seed is not None and request.seed < 0:
            raise OptionsValidationError(
                f"seed must be non-negative, got {request.seed}"
            )

    @staticmethod
    def _static_request(request: AnimationRequest) -> GenerationRequest:
        """Build the request used only for static recipe construction."""
        return GenerationRequest(
            size=request.size,
            generator_name=request.generator_name,
            output_path=request.output_path,
            seed=request.seed,
            overwrite=request.overwrite,
            options=GenerationOptions(
                palette_name=request.options.palette_name,
            ),
        )

    @staticmethod
    def _rarity_is_accepted(
        *,
        actual_tier: str,
        minimum_tier: str | None,
    ) -> bool:
        """Return whether the actual rarity satisfies an optional minimum."""
        if minimum_tier is None:
            return True
        return _RARITY_RANK[actual_tier] >= _RARITY_RANK[minimum_tier]


class _CandidateResult:
    """Internal holder for one fully rendered animation candidate."""

    __slots__ = (
        "animation_recipe",
        "animation_trait_probs",
        "animation_rarity",
        "frames",
        "temporal_quality",
    )

    def __init__(
        self,
        *,
        animation_recipe: AnimationRecipe,
        animation_trait_probs: list[TraitProbability],
        animation_rarity: RarityResult,
        frames: list[UInt8Array],
        temporal_quality: TemporalQualityResult,
    ) -> None:
        self.animation_recipe = animation_recipe
        self.animation_trait_probs = animation_trait_probs
        self.animation_rarity = animation_rarity
        self.frames = frames
        self.temporal_quality = temporal_quality