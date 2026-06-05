"""CLI command: pixelforge animate

Generates a seamless looping GIF from a procedural recipe generator.
Every frame is rendered mathematically — this is not a post-processing tool.

Example:
    pixelforge animate --generator harmonic-waves --width 512 --height 512 \\
        --seed 42 --frames 48 --fps 24 --output output/harmonic-waves.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pixel_forge.core.models.animation_options import AnimationOptions
from pixel_forge.core.models.animation_request import AnimationRequest
from pixel_forge.core.models.image_size import ImageSize
from pixel_forge.services.animation_service import AnimationService


def configure_animate_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Register the `animate` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "animate",
        help="Generate a seamless looping GIF from a procedural recipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    p.add_argument("--generator", "-g", default="harmonic-waves",
                   help="Generator name (default: harmonic-waves)")
    p.add_argument("--width", "-W", type=int, default=512,
                   help="Frame width in pixels (default: 512)")
    p.add_argument("--height", "-H", type=int, default=512,
                   help="Frame height in pixels (default: 512)")
    p.add_argument("--seed", "-s", type=int, default=None,
                   help="Master seed for deterministic output (default: random)")
    p.add_argument("--frames", "-f", type=int, default=24,
                   help="Number of frames in the GIF (default: 24)")
    p.add_argument("--fps", type=int, default=24,
                   help="Frames per second (default: 24)")
    p.add_argument("--loop-count", type=int, default=0,
                   help="Loop count: 0 = infinite (default: 0)")
    p.add_argument("--motion-profile", "-m", default=None,
                   help="Motion profile override (generator-specific)")
    p.add_argument("--motion-intensity", type=float, default=None,
                   help="Motion intensity 0.0–1.0 (default: auto)")
    p.add_argument("--colors", type=int, default=256, choices=[64, 128, 256],
                   help="GIF palette size: 64, 128, or 256 (default: 256)")
    p.add_argument("--dither", default="none", choices=["none", "floyd-steinberg"],
                   help="GIF dithering mode (default: none)")
    p.add_argument("--temporal-quality-threshold", type=float, default=None,
                   help="Minimum temporal quality score 0.0–1.0")
    p.add_argument("--max-animation-retries", type=int, default=3,
                   help="Maximum animation retry attempts (default: 3)")
    p.add_argument("--strict-temporal-quality", action="store_true",
                   help="Fail if all retries are rejected by temporal quality")
    p.add_argument("--no-metadata", action="store_true",
                   help="Skip writing the JSON manifest")
    p.add_argument("--palette", default=None,
                   help="Force a specific palette name")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--output", "-o", required=True,
                   help="Output .gif file path")


def run_animate_command(
    args: argparse.Namespace,
    service: AnimationService,
) -> int:
    """Execute the animate command. Returns a process exit code."""
    opts = AnimationOptions(
        frame_count=args.frames,
        fps=args.fps,
        loop_count=args.loop_count,
        motion_profile=args.motion_profile,
        motion_intensity=args.motion_intensity,
        temporal_quality_threshold=args.temporal_quality_threshold,
        max_animation_retries=args.max_animation_retries,
        write_metadata=not args.no_metadata,
        gif_colors=args.colors,
        gif_dither=args.dither,
        strict_temporal_quality=args.strict_temporal_quality,
        palette_name=args.palette,
    )

    request = AnimationRequest(
        size=ImageSize(width=args.width, height=args.height),
        generator_name=args.generator,
        output_path=Path(args.output),
        seed=args.seed,
        overwrite=args.overwrite,
        options=opts,
    )

    result = service.animate(request)

    print(f"Generated: {result.output_path}")
    print(f"  Generator:     {result.generator_name}")
    print(f"  Dimensions:    {result.size.width}×{result.size.height}")
    print(f"  Frames:        {result.frame_count} @ {result.fps} fps")
    print(f"  Duration:      {result.frame_count * result.frame_duration_ms} ms")
    print(f"  Motion:        {result.motion_profile}")
    print(f"  Seed:          {result.master_seed}")
    print(f"  Animation seed:{result.animation_seed}")
    print(f"  Rarity:        {result.animation_rarity_tier}")
    print(f"  Temporal Q:    {result.temporal_quality_score:.3f}")
    print(f"  Size:          {result.bytes_written:,} bytes")
    print(f"  Content ID:    {result.content_id[:16]}...")
    if result.metadata_path:
        print(f"  Manifest:      {result.metadata_path}")
    return 0
