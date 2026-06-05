"""PixelForge CLI entry point and dependency composition."""

import sys
from collections.abc import Sequence

from pixel_forge.cli.commands.explore import run_explore_command
from pixel_forge.cli.commands.generate import run_generate_command
from pixel_forge.cli.commands.inspect_seed import run_inspect_seed_command
from pixel_forge.cli.commands.list_generators import run_list_generators_command
from pixel_forge.cli.parser import build_parser
from pixel_forge.core.config import Settings
from pixel_forge.core.exceptions import PixelForgeError
from pixel_forge.generators.registry import build_default_registry
from pixel_forge.image.encoders import PngEncoder
from pixel_forge.image.writers import AtomicFileWriter
from pixel_forge.services import GenerationService
from pixel_forge.shared.validation import RequestValidator


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    settings = Settings()
    registry = build_default_registry()
    parser = build_parser(settings)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "generate":
            service = GenerationService(
                registry=registry,
                validator=RequestValidator(settings),
                encoder=PngEncoder(compress_level=settings.png_compress_level),
                writer=AtomicFileWriter(),
                supported_output_suffixes=settings.supported_output_suffixes,
            )
            return run_generate_command(arguments, service)

        if arguments.command == "list-generators":
            return run_list_generators_command(registry)

        if arguments.command == "inspect-seed":
            return run_inspect_seed_command(arguments, registry)

        if arguments.command == "explore":
            return run_explore_command(arguments, registry, settings)

        parser.error(f"Unsupported command: {arguments.command}")
    except PixelForgeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    return 0
