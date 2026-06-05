"""Application service that coordinates procedural GIF animation generation.

Pipeline:
  1.  Validate the animation request and options.
  2.  Resolve the static generator (must be a RecipeGenerator).
  3.  Resolve the animated renderer (must be an AnimatedRecipeRenderer).
  4.  Build the static artwork recipe (same pipeline as GenerationService).
  5.  Apply static compatibility rules to the recipe.
  6.  Derive a deterministic animation seed from the static candidate seed.
  7.  Build the animation recipe (all motion decisions, no pixels yet).
  8.  Render representative frames for palette construction and quality probing.
  9.  Build a deterministic global GIF palette from representative frames.
 10.  Render all N frames and collect them.
 11.  Evaluate temporal quality.
 12.  Retry deterministically if temporal quality is below threshold.
 13.  Encode the GIF from palette-indexed frames.
 14.  Write GIF and JSON manifest atomically.
 15.  Return AnimationResult.

Static PNG output is NOT modified by this service.
"""

from __future__ import annotations

import secrets
import tempfile
import os
from pathlib import Path

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.aesthetics.palettes.palette_registry import build_default_palette_registry
from pixel_forge.animation.animation_randomness import (
    AnimationStreams,
    derive_animation_retry_seed,
    derive_animation_seed,
)
from pixel_forge.animation.frame_phase import generate_frame_phases, representative_phases
from pixel_forge.animation.motion_profiles import (
    ALL_PROFILES_BY_GENERATOR,
    DEFAULT_PROFILE_BY_GENERATOR,
)
from pixel_forge.animation.protocols import AnimatedRecipeRenderer
from pixel_forge.animation.quality import TemporalQualityEvaluator, TemporalQualityThresholds
from pixel_forge.core.exceptions import (
    GeneratorNotFoundError,
    OptionsValidationError,
    QualityRejectionError,
    UnsupportedOutputFormatError,
    ValidationError,
)
from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_recipe import AnimationRecipe
from pixel_forge.core.models.animation_request import AnimationRequest
from pixel_forge.core.models.animation_result import AnimationResult
from pixel_forge.core.models.image_size import ImageSize
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

_PALETTE_REGISTRY = build_default_palette_registry()

# Maps generator name → animated renderer class.
_ANIMATED_RENDERERS: dict[str, type[AnimatedRecipeRenderer]] = {}


def _build_renderer_registry() -> dict[str, AnimatedRecipeRenderer]:
    """Import and instantiate all animated renderers."""
    from pixel_forge.generators.harmonic_waves.animation import HarmonicWavesAnimator
    from pixel_forge.generators.radial_bloom.animation import RadialBloomAnimator
    from pixel_forge.generators.plasma_flow.animation import PlasmaFlowAnimator
    from pixel_forge.generators.mandelbrot_dream.animation import MandelbrotDreamAnimator

    renderers: list[AnimatedRecipeRenderer] = [
        HarmonicWavesAnimator(),
        RadialBloomAnimator(),
        PlasmaFlowAnimator(),
        MandelbrotDreamAnimator(),
    ]
    return {r.name: r for r in renderers}


_RENDERER_REGISTRY: dict[str, AnimatedRecipeRenderer] = _build_renderer_registry()

# Safety limits
_MIN_FRAMES = 2
_MAX_FRAMES = 120
_MIN_FPS = 1
_MAX_FPS = 60
_MAX_PIXEL_BUDGET = 512 * 512 * 120  # width × height × frame_count


class AnimationService:
    """Execute one complete, validated animation generation operation."""

    def __init__(
        self,
        *,
        registry: GeneratorRegistry,
        writer: AtomicFileWriter | None = None,
        temporal_quality_thresholds: TemporalQualityThresholds | None = None,
    ) -> None:
        self._registry = registry
        self._writer = writer or AtomicFileWriter()
        self._compat_validator = RecipeCompatibilityValidator()
        self._rarity_evaluator = RarityEvaluator()
        self._temporal_evaluator = TemporalQualityEvaluator(temporal_quality_thresholds)
        self._manifest_writer = AnimationManifestWriter()

    def animate(self, request: AnimationRequest) -> AnimationResult:
        """Generate, validate, encode, and persist one animated GIF."""
        self._validate_request(request)

        output_path = request.output_path
        if output_path.suffix.lower() != ".gif":
            raise UnsupportedOutputFormatError(
                f"Animation output must be a .gif file, got: {output_path.suffix}"
            )
        if output_path.exists() and not request.overwrite:
            from pixel_forge.core.exceptions import OutputFileExistsError
            raise OutputFileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite to replace."
            )

        opts = request.options
        generator = self._registry.get(request.generator_name)
        if not isinstance(generator, RecipeGenerator):
            raise ValidationError(
                f"Generator '{request.generator_name}' does not support animation "
                "(it is not a recipe-driven generator)."
            )

        renderer = _RENDERER_REGISTRY.get(request.generator_name)
        if renderer is None:
            raise ValidationError(
                f"Generator '{request.generator_name}' does not have an animated "
                "renderer. Supported: " + ", ".join(sorted(_RENDERER_REGISTRY))
            )

        master_seed = request.seed if request.seed is not None else secrets.randbits(64)
        schema_version = "1.0"

        # Build static recipe (retry 0 only — animation quality drives retries).
        candidate_seed = derive_candidate_seed(
            master_seed=master_seed,
            generator_name=generator.name,
            retry_index=0,
            schema_version=schema_version,
        )
        static_streams = RandomStreams.from_seed(candidate_seed)
        static_recipe, static_trait_probs = generator.build_recipe(
            self._static_request(request),
            static_streams,
            candidate_seed=candidate_seed,
            retry_index=0,
        )
        compat_result = self._compat_validator.validate(static_recipe)
        static_recipe = compat_result.recipe
        applied_rules = compat_result.applied_rules
        static_rarity = self._rarity_evaluator.evaluate(static_trait_probs)

        # Temporal quality retry loop.
        max_retries = opts.max_animation_retries
        best_result: _CandidateResult | None = None
        best_score = -1.0
        accepted: _CandidateResult | None = None

        for retry_index in range(max_retries + 1):
            animation_seed = derive_animation_seed(
                candidate_seed=candidate_seed,
                generator_name=generator.name,
                animation_schema_version="1.0",
            )
            if retry_index > 0:
                animation_seed = derive_animation_retry_seed(
                    animation_seed=animation_seed,
                    retry_index=retry_index,
                )

            anim_streams = AnimationStreams.from_seed(animation_seed)
            animation_recipe, anim_trait_probs = renderer.build_animation_recipe(
                static_recipe, opts, anim_streams, animation_seed
            )
            animation_rarity = self._rarity_evaluator.evaluate(anim_trait_probs)

            frames = self._render_all_frames(renderer, animation_recipe)
            virtual_p1 = renderer.render_frame(animation_recipe, 0.0)
            temporal_quality = self._temporal_evaluator.evaluate(
                frames, virtual_phase1=virtual_p1
            )

            if temporal_quality.aggregate_score > best_score:
                best_score = temporal_quality.aggregate_score
                best_result = _CandidateResult(
                    animation_recipe=animation_recipe,
                    anim_trait_probs=anim_trait_probs,
                    animation_rarity=animation_rarity,
                    frames=frames,
                    temporal_quality=temporal_quality,
                )

            if temporal_quality.accepted:
                accepted = best_result
                break

        if opts.strict_temporal_quality and accepted is None:
            reasons = best_result.temporal_quality.rejection_reasons if best_result else ()
            raise QualityRejectionError(
                f"All {max_retries + 1} animation candidates failed temporal quality. "
                + " ".join(reasons)
            )

        final = accepted if accepted is not None else best_result
        if final is None:
            raise RuntimeError("Animation service produced no candidate result.")

        # Encode GIF.
        gif_colors = min(max(opts.gif_colors, 2), 256)
        gif_opts = GifEncodingOptions(
            gif_colors=gif_colors,
            dither=opts.gif_dither,
            loop_count=opts.loop_count,
        )
        encoder = GifEncoder(gif_opts)
        gif_bytes = encoder.encode(
            final.frames,
            frame_duration_ms=final.animation_recipe.frame_duration_ms,
        )

        # Compute content ID.
        content_id = compute_animation_content_id(final.animation_recipe, final.frames)

        # Write GIF atomically.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path = self._write_gif(gif_bytes, output_path, overwrite=request.overwrite)

        # Write manifest.
        metadata_path: Path | None = None
        if opts.write_metadata:
            manifest = build_animation_manifest(
                animation_recipe=final.animation_recipe,
                static_rarity=static_rarity,
                animation_rarity=final.animation_rarity,
                temporal_quality=final.temporal_quality,
                applied_rules=applied_rules,
                gif_path=gif_path,
                gif_colors=gif_colors,
                gif_dither=opts.gif_dither,
                content_id=content_id,
            )
            metadata_path = self._manifest_writer.write(
                manifest, gif_path, overwrite=request.overwrite
            )

        return AnimationResult(
            output_path=gif_path,
            size=ImageSize(width=static_recipe.width, height=static_recipe.height),
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

    # ──────────────────────────────────────────────────────────────────────────

    def _render_all_frames(
        self,
        renderer: AnimatedRecipeRenderer,
        animation_recipe: AnimationRecipe,
    ) -> list[UInt8Array]:
        phases = generate_frame_phases(animation_recipe.frame_count)
        return [
            np.ascontiguousarray(renderer.render_frame(animation_recipe, p))
            for p in phases
        ]

    def _write_gif(
        self,
        gif_bytes: bytes,
        output_path: Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Write GIF bytes atomically, return the final path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".gif.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(gif_bytes)
                f.flush()
                os.fsync(f.fileno())
            if output_path.exists() and overwrite:
                output_path.unlink()
            os.replace(tmp, output_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return output_path

    def _validate_request(self, request: AnimationRequest) -> None:
        opts = request.options
        if opts.frame_count < _MIN_FRAMES:
            raise OptionsValidationError(
                f"frame_count must be >= {_MIN_FRAMES}, got {opts.frame_count}"
            )
        if opts.frame_count > _MAX_FRAMES:
            raise OptionsValidationError(
                f"frame_count must be <= {_MAX_FRAMES}, got {opts.frame_count}"
            )
        if opts.fps < _MIN_FPS:
            raise OptionsValidationError(
                f"fps must be >= {_MIN_FPS}, got {opts.fps}"
            )
        if opts.fps > _MAX_FPS:
            raise OptionsValidationError(
                f"fps must be <= {_MAX_FPS}, got {opts.fps}"
            )
        if opts.gif_colors not in (64, 128, 256):
            raise OptionsValidationError(
                f"gif_colors must be 64, 128, or 256, got {opts.gif_colors}"
            )
        if opts.gif_dither not in ("none", "floyd-steinberg"):
            raise OptionsValidationError(
                f"gif_dither must be 'none' or 'floyd-steinberg', got {opts.gif_dither!r}"
            )
        pixel_budget = (
            request.size.width * request.size.height * opts.frame_count
        )
        if pixel_budget > _MAX_PIXEL_BUDGET:
            raise OptionsValidationError(
                f"Animation pixel budget {pixel_budget:,} exceeds maximum "
                f"{_MAX_PIXEL_BUDGET:,} (reduce width, height, or frame_count)."
            )
        if opts.motion_profile is not None:
            valid = ALL_PROFILES_BY_GENERATOR.get(request.generator_name, ())
            if opts.motion_profile not in valid:
                valid_str = ", ".join(valid) if valid else "(none)"
                raise OptionsValidationError(
                    f"motion_profile {opts.motion_profile!r} is not valid for "
                    f"'{request.generator_name}'. Valid: {valid_str}"
                )
        if opts.loop_count < 0:
            raise OptionsValidationError(
                f"loop_count must be >= 0 (0 = infinite), got {opts.loop_count}"
            )
        if opts.max_animation_retries < 0:
            raise OptionsValidationError(
                f"max_animation_retries must be >= 0, got {opts.max_animation_retries}"
            )

    @staticmethod
    def _static_request(request: AnimationRequest) -> object:
        """Build a minimal GenerationRequest for static recipe building."""
        from pixel_forge.core.models.generation_options import GenerationOptions
        from pixel_forge.core.models.generation_request import GenerationRequest

        return GenerationRequest(
            size=request.size,
            generator_name=request.generator_name,
            output_path=request.output_path,
            seed=request.seed,
            options=GenerationOptions(
                palette_name=request.options.palette_name,
            ),
        )


class _CandidateResult:
    """Internal holder for one retry candidate's outputs."""

    __slots__ = (
        "animation_recipe",
        "anim_trait_probs",
        "animation_rarity",
        "frames",
        "temporal_quality",
    )

    def __init__(
        self,
        animation_recipe: AnimationRecipe,
        anim_trait_probs: list,
        animation_rarity: object,
        frames: list[UInt8Array],
        temporal_quality: object,
    ) -> None:
        self.animation_recipe = animation_recipe
        self.anim_trait_probs = anim_trait_probs
        self.animation_rarity = animation_rarity  # type: ignore[assignment]
        self.frames = frames
        self.temporal_quality = temporal_quality  # type: ignore[assignment]
