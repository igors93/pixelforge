"""Implementation of the ``explore`` CLI command.

Generates a batch of deterministic candidates, evaluates each one, keeps the
highest-scoring results, saves PNG and JSON for kept images, and creates a
contact sheet summary. Image arrays are not kept in memory simultaneously;
they are rendered, evaluated, encoded, and written one at a time.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.image.encoders import PngEncoder
from pixel_forge.image.writers import AtomicFileWriter
from pixel_forge.services import GenerationService
from pixel_forge.shared.validation import RequestValidator


@dataclass
class _CandidateSummary:
    index: int
    seed: int
    quality_score: float
    rarity_tier: str
    png_path: Path | None
    metadata_path: Path | None


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

    encoder = PngEncoder(compress_level=settings.png_compress_level)
    writer = AtomicFileWriter()
    service = GenerationService(
        registry=registry,
        validator=RequestValidator(settings),
        encoder=encoder,
        writer=writer,
        supported_output_suffixes=settings.supported_output_suffixes,
    )

    print(
        f"Exploring {count} candidates for '{arguments.generator}' "
        f"(master seed {master_seed}) …"
    )

    summaries: list[_CandidateSummary] = []

    for i in range(count):
        # Derive a deterministic candidate seed for each exploration slot.
        import hashlib
        import struct

        payload = f"explore|{master_seed}|{arguments.generator}|{i}".encode()
        digest = hashlib.sha256(payload).digest()
        candidate_seed = struct.unpack_from(">Q", digest)[0]

        options = GenerationOptions(
            palette_name=arguments.palette,
            write_metadata=False,  # Don't write until we know if it's kept.
            max_retries=0,         # No retries in exploration; just evaluate.
        )
        request = GenerationRequest(
            size=ImageSize(width=arguments.width, height=arguments.height),
            generator_name=arguments.generator,
            output_path=output_dir / f"_probe_{i}.png",
            seed=candidate_seed,
            overwrite=True,
            options=options,
        )
        try:
            result = service.generate(request)
            summaries.append(
                _CandidateSummary(
                    index=i,
                    seed=candidate_seed,
                    quality_score=result.quality_score,
                    rarity_tier=result.overall_rarity_tier,
                    png_path=None,
                    metadata_path=None,
                )
            )
            # Remove the probe file; we'll re-render the kept ones below.
            probe_path = output_dir / f"_probe_{i}.png"
            if probe_path.exists():
                probe_path.unlink()
        except Exception as exc:
            print(f"  Candidate {i}: error – {exc}", file=sys.stderr)

    if not summaries:
        print("No valid candidates were produced.", file=sys.stderr)
        return 2

    # Rank candidates by the chosen criterion.
    tier_order = {"Common": 0, "Uncommon": 1, "Rare": 2, "Epic": 3, "Legendary": 4}

    def score_key(s: _CandidateSummary) -> float:
        rarity_score = tier_order.get(s.rarity_tier, 0) / 4.0
        if sort_by == "quality":
            return s.quality_score
        elif sort_by == "rarity":
            return rarity_score
        else:  # balanced
            return 0.6 * s.quality_score + 0.4 * rarity_score

    ranked = sorted(summaries, key=score_key, reverse=True)
    top_candidates = ranked[:keep]

    print(f"\nKeeping {len(top_candidates)} of {len(summaries)} evaluated candidates:")

    kept_paths: list[Path] = []
    for rank, candidate in enumerate(top_candidates):
        filename = (
            f"{arguments.generator}-seed{candidate.seed}"
            f"-q{candidate.quality_score:.3f}"
            f"-{candidate.rarity_tier.lower()}.png"
        )
        png_path = output_dir / filename
        options = GenerationOptions(
            palette_name=arguments.palette,
            write_metadata=True,
            max_retries=0,
        )
        request = GenerationRequest(
            size=ImageSize(width=arguments.width, height=arguments.height),
            generator_name=arguments.generator,
            output_path=png_path,
            seed=candidate.seed,
            overwrite=True,
            options=options,
        )
        try:
            result = service.generate(request)
            candidate.png_path = result.output_path
            candidate.metadata_path = result.metadata_path
            kept_paths.append(result.output_path)
            print(
                f"  [{rank + 1}] seed={candidate.seed} "
                f"quality={candidate.quality_score:.3f} "
                f"rarity={candidate.rarity_tier} → {result.output_path.name}"
            )
        except Exception as exc:
            print(f"  [{rank + 1}] Error saving candidate: {exc}", file=sys.stderr)

    # Contact sheet.
    if not arguments.no_contact_sheet and kept_paths:
        _build_contact_sheet(kept_paths, output_dir, arguments.generator)

    # Exploration summary JSON.
    _write_summary(
        output_dir=output_dir,
        generator=arguments.generator,
        master_seed=master_seed,
        count=count,
        sort_by=sort_by,
        summaries=ranked[:keep],
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
        sheet_w = cols * thumb_w
        sheet_h = rows * thumb_h

        sheet = PILImage.new("RGB", (sheet_w, sheet_h))
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
    summaries: list[_CandidateSummary],
) -> None:
    data: dict[str, Any] = {
        "generator": generator,
        "master_seed": master_seed,
        "total_generated": count,
        "kept": len(summaries),
        "sort_by": sort_by,
        "candidates": [
            {
                "index": s.index,
                "seed": s.seed,
                "quality_score": s.quality_score,
                "rarity_tier": s.rarity_tier,
                "png": s.png_path.name if s.png_path else None,
                "metadata": s.metadata_path.name if s.metadata_path else None,
            }
            for s in summaries
        ],
    }
    summary_path = output_dir / f"{generator}-exploration-summary.json"
    summary_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Exploration summary: {summary_path}")
