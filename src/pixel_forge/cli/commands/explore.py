"""Implementation of the ``explore`` CLI command.

Generates a batch of deterministic candidates, evaluates each one without writing
to disk, keeps the highest-scoring results in a bounded in-memory top-K cache,
then writes ONLY the kept images. This avoids the previous double-render bug
where probes were written, deleted, and then re-rendered for final output.

Memory usage is bounded: only the top-K RGB arrays are retained simultaneously.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.aesthetics.quality.quality_evaluator import QualityEvaluator, QualityThresholds
from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.core.models.artwork_recipe import ArtworkRecipe
from pixel_forge.generators.common.recipe_generator import RecipeGenerator
from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.image.encoders import PngEncoder
from pixel_forge.image.writers import AtomicFileWriter
from pixel_forge.metadata.artwork_manifest import build_manifest
from pixel_forge.metadata.manifest_writer import ManifestWriter
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.rarity_evaluator import RarityEvaluator
from pixel_forge.rarity.rarity_tier import RarityTier
from pixel_forge.shared.validation import RequestValidator


@dataclass
class _EvaluatedCandidate:
    """One fully-evaluated candidate held in memory until save/discard."""

    index: int
    seed: int
    quality_score: float
    rarity_tier: str
    rarity_bits: float
    recipe: ArtworkRecipe
    rgb: UInt8Array
    applied_rules: tuple[str, ...]
    rarity_result: Any  # RarityResult
    quality_result: Any  # QualityResult
    png_path: Path | None = field(default=None)
    metadata_path: Path | None = field(default=None)

    def ranking_score(self, sort_by: str) -> float:
        tier_order = {t.value: i for i, t in enumerate(RarityTier)}
        rarity_score = tier_order.get(self.rarity_tier, 0) / (len(tier_order) - 1)
        if sort_by == "quality":
            return self.quality_score
        if sort_by == "rarity":
            return self.rarity_bits  # use raw bits for stable ranking
        return 0.6 * self.quality_score + 0.4 * rarity_score


def configure_explore_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    settings: Settings,
) -> None:
    parser = subparsers.add_parser(
        "explore",
        help="Generate many candidates and keep the best results.",
    )
    parser.add_argument(
        "--generator",
        default=settings.default_generator,
        help=f"Generator name (default: {settings.default_generator}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        metavar="INTEGER",
        help="Total number of candidates to generate (default: 20).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        metavar="INTEGER",
        help="Number of top results to save (default: 5).",
    )
    parser.add_argument(
        "--width", type=int, default=settings.default_width, help="Canvas width."
    )
    parser.add_argument(
        "--height", type=int, default=settings.default_height, help="Canvas height."
    )
    parser.add_argument(
        "--sort-by",
        default="quality",
        choices=["quality", "rarity", "balanced"],
        help="Ranking criterion for keeping candidates (default: quality).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/exploration"),
        help="Output directory for kept images (default: output/exploration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for candidate sequence determinism.",
    )
    parser.add_argument(
        "--palette", default=None, metavar="PALETTE_NAME", help="Force a named palette."
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Skip generating the PNG contact sheet.",
    )


def run_explore_command(
    arguments: argparse.Namespace,
    registry: GeneratorRegistry,
    settings: Settings,
) -> int:
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_seed = arguments.seed if arguments.seed is not None else secrets.randbits(64)
    count = arguments.count
    keep = min(arguments.keep, count)
    sort_by: str = arguments.sort_by

    generator = registry.get(arguments.generator)
    if not isinstance(generator, RecipeGenerator):
        print(
            f"Error: '{arguments.generator}' does not support recipe exploration.",
            file=sys.stderr,
        )
        return 2

    encoder = PngEncoder(compress_level=settings.png_compress_level)
    compat_validator = RecipeCompatibilityValidator()
    quality_evaluator = QualityEvaluator(QualityThresholds())
    rarity_evaluator = RarityEvaluator()

    print(
        f"Exploring {count} candidates for '{arguments.generator}' "
        f"(master seed {master_seed}) …"
    )

    # Bounded top-K heap: keeps only the best `keep` candidates in memory.
    candidates: list[_EvaluatedCandidate] = []

    for i in range(count):
        payload = f"explore|{master_seed}|{arguments.generator}|{i}".encode()
        digest = hashlib.sha256(payload).digest()
        candidate_seed: int = struct.unpack_from(">Q", digest)[0]

        options = GenerationOptions(
            palette_name=arguments.palette,
            write_metadata=False,
            max_retries=0,
        )
        request = GenerationRequest(
            size=ImageSize(width=arguments.width, height=arguments.height),
            generator_name=arguments.generator,
            output_path=output_dir / "_unused.png",  # never written in eval phase
            seed=candidate_seed,
            overwrite=True,
            options=options,
        )

        try:
            eff_seed = derive_candidate_seed(
                master_seed=candidate_seed,
                generator_name=generator.name,
                retry_index=0,
                schema_version="1.0",
            )
            streams = RandomStreams.from_seed(eff_seed)
            recipe, trait_probs = generator.build_recipe(
                request, streams, candidate_seed=eff_seed, retry_index=0
            )
            compat_result = compat_validator.validate(recipe)
            recipe = compat_result.recipe

            rgb: UInt8Array = np.ascontiguousarray(generator.render_recipe(recipe))
            quality = quality_evaluator.evaluate(rgb)
            rarity = rarity_evaluator.evaluate(trait_probs)

            candidate = _EvaluatedCandidate(
                index=i,
                seed=candidate_seed,
                quality_score=quality.aggregate_score,
                rarity_tier=rarity.overall_tier,
                rarity_bits=rarity.total_information_bits,
                recipe=recipe,
                rgb=rgb,
                applied_rules=compat_result.applied_rules,
                rarity_result=rarity,
                quality_result=quality,
            )

            # Bounded insert: keep only top `keep` by ranking score.
            candidates.append(candidate)
            candidates.sort(key=lambda c: c.ranking_score(sort_by), reverse=True)
            if len(candidates) > keep:
                # Drop the worst (last after sort) to bound memory.
                candidates.pop()

        except Exception as exc:
            print(f"  Candidate {i}: error – {exc}", file=sys.stderr)

    if not candidates:
        print("No valid candidates were produced.", file=sys.stderr)
        return 2

    print(f"\nKeeping {len(candidates)} of {count} evaluated candidates:")

    kept_paths: list[Path] = []
    manifest_writer = ManifestWriter()
    writer = AtomicFileWriter()

    for rank, candidate in enumerate(candidates):
        filename = (
            f"{arguments.generator}-seed{candidate.seed}"
            f"-q{candidate.quality_score:.3f}"
            f"-{candidate.rarity_tier.lower()}.png"
        )
        png_path = output_dir / filename

        try:
            # Encode and write the already-computed RGB array (no re-render).
            from pixel_forge.core.models import GeneratedImage

            image = GeneratedImage(
                size=ImageSize(width=candidate.recipe.width, height=candidate.recipe.height),
                pixels=candidate.rgb.tobytes(),
                generator_name=generator.name,
                seed=candidate.seed,
            )
            encoded = encoder.encode(image)
            final_path = writer.write(encoded, png_path, overwrite=True)
            candidate.png_path = final_path

            manifest = build_manifest(
                recipe=candidate.recipe,
                rarity=candidate.rarity_result,
                quality=candidate.quality_result,
                applied_rules=candidate.applied_rules,
                png_path=final_path,
                rgb_array=candidate.rgb,
            )
            candidate.metadata_path = manifest_writer.write(
                manifest, final_path, overwrite=True
            )
            kept_paths.append(final_path)
            print(
                f"  [{rank + 1}] seed={candidate.seed} "
                f"quality={candidate.quality_score:.3f} "
                f"rarity={candidate.rarity_tier} "
                f"bits={candidate.rarity_bits:.1f} → {final_path.name}"
            )
        except Exception as exc:
            print(f"  [{rank + 1}] Error saving candidate: {exc}", file=sys.stderr)

    if not arguments.no_contact_sheet and kept_paths:
        _build_contact_sheet(kept_paths, output_dir, arguments.generator)

    _write_summary(
        output_dir=output_dir,
        generator=arguments.generator,
        master_seed=master_seed,
        count=count,
        sort_by=sort_by,
        candidates=candidates,
    )

    print(f"\nExploration complete. Results saved to: {output_dir}")
    return 0


def _build_contact_sheet(paths: list[Path], output_dir: Path, generator: str) -> None:
    try:
        from PIL import Image as PILImage

        images = []
        for p in paths:
            if p.exists():
                images.append(PILImage.open(p))
        if not images:
            return

        cols = min(len(images), 4)
        rows = (len(images) + cols - 1) // cols
        thumb_w, thumb_h = images[0].width, images[0].height
        sheet = PILImage.new("RGB", (cols * thumb_w, rows * thumb_h))
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            sheet.paste(img, (col * thumb_w, row * thumb_h))
            img.close()

        sheet_path = output_dir / f"{generator}-contact-sheet.png"
        sheet.save(sheet_path, format="PNG", compress_level=6)
        sheet.close()
        print(f"Contact sheet      : {sheet_path}")
    except Exception as exc:
        print(f"Could not build contact sheet: {exc}", file=sys.stderr)


def _write_summary(
    *,
    output_dir: Path,
    generator: str,
    master_seed: int,
    count: int,
    sort_by: str,
    candidates: list[_EvaluatedCandidate],
) -> None:
    data: dict[str, Any] = {
        "generator": generator,
        "master_seed": master_seed,
        "total_generated": count,
        "kept": len(candidates),
        "sort_by": sort_by,
        "candidates": [
            {
                "index": c.index,
                "seed": c.seed,
                "quality_score": c.quality_score,
                "rarity_tier": c.rarity_tier,
                "rarity_bits": c.rarity_bits,
                "png": c.png_path.name if c.png_path else None,
                "metadata": c.metadata_path.name if c.metadata_path else None,
            }
            for c in candidates
        ],
    }
    summary_path = output_dir / f"{generator}-exploration-summary.json"
    summary_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Exploration summary: {summary_path}")


def _make_request(
    *,
    generator: str,
    size: ImageSize,
    seed: int,
    output_dir: Path,
    options: GenerationOptions,
) -> GenerationRequest:
    return GenerationRequest(
        size=size,
        generator_name=generator,
        output_path=output_dir / "_unused.png",
        seed=seed,
        overwrite=True,
        options=options,
    )


# Keep RequestValidator import so it can be used by callers that need it.
__all__ = [
    "configure_explore_command",
    "run_explore_command",
]
_validator_cls = RequestValidator  # noqa: F401 – keep import live
