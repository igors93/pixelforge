"""Implementation of the ``generate`` CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.services import GenerationService


def configure_generate_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    settings: Settings,
) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Generate and save one procedural image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=settings.default_width,
        help=f"Image width in pixels (maximum: {settings.max_width}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=settings.default_height,
        help=f"Image height in pixels (maximum: {settings.max_height}).",
    )
    parser.add_argument(
        "--generator",
        default=settings.default_generator,
        help=f"Registered generator name (default: {settings.default_generator}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.default_output_path,
        help=f"Destination PNG path (default: {settings.default_output_path}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional non-negative seed for reproducible output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file when it already exists.",
    )
    # New options added in v0.3.
    parser.add_argument(
        "--palette",
        default=None,
        metavar="PALETTE_NAME",
        help="Force a specific named palette (e.g. ocean-depth, solar-flare).",
    )
    parser.add_argument(
        "--complexity",
        default=None,
        choices=["minimal", "simple", "moderate", "complex", "intricate"],
        help="Override complexity level.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum aggregate quality score [0.0–1.0] before retry.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        metavar="INTEGER",
        help="Maximum quality retry attempts (default: 5).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Suppress writing the JSON manifest beside the PNG.",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Fail with exit code 2 when all retry candidates are rejected.",
    )


def run_generate_command(
    arguments: argparse.Namespace,
    service: GenerationService,
) -> int:
    options = GenerationOptions(
        palette_name=arguments.palette,
        complexity_level=arguments.complexity,
        quality_threshold=arguments.quality_threshold,
        max_retries=arguments.max_retries,
        write_metadata=not arguments.no_metadata,
        strict_quality=arguments.strict_quality,
    )
    request = GenerationRequest(
        size=ImageSize(width=arguments.width, height=arguments.height),
        generator_name=arguments.generator,
        output_path=arguments.output,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
        options=options,
    )
    result = service.generate(request)

    print("Image generated successfully.")
    print(f"Generator : {result.generator_name}")
    print(f"Size      : {result.size.width}x{result.size.height} pixels")
    print(f"Seed      : {result.seed}")
    print(f"Rarity    : {result.overall_rarity_tier}")
    print(f"Quality   : {result.quality_score:.3f}")
    print(f"Retries   : {result.retry_index}")
    print(f"Output    : {result.output_path}")
    if result.metadata_path:
        print(f"Metadata  : {result.metadata_path}")
    print(f"File size : {result.bytes_written} bytes")
    return 0
