"""Argument parser construction for the PixelForge CLI."""

import argparse

from pixel_forge import __version__
from pixel_forge.cli.commands.generate import configure_generate_command
from pixel_forge.cli.commands.list_generators import configure_list_generators_command
from pixel_forge.core.config import Settings


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixelforge",
        description="Generate procedural images from the command line.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"PixelForge {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    configure_generate_command(subparsers, settings)
    configure_list_generators_command(subparsers)
    return parser
