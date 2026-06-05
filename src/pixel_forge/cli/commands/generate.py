"""Implementation of the ``generate`` CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationRequest, ImageSize
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


def run_generate_command(
    arguments: argparse.Namespace,
    service: GenerationService,
) -> int:
    request = GenerationRequest(
        size=ImageSize(width=arguments.width, height=arguments.height),
        generator_name=arguments.generator,
        output_path=arguments.output,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )
    result = service.generate(request)

    print("Image generated successfully.")
    print(f"Generator: {result.generator_name}")
    print(f"Size: {result.size.width}x{result.size.height} pixels")
    print(f"Seed: {result.seed}")
    print(f"Output: {result.output_path}")
    print(f"File size: {result.bytes_written} bytes")
    return 0
