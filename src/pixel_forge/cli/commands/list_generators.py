"""Implementation of the ``list-generators`` CLI command."""

from __future__ import annotations

import argparse

from pixel_forge.generators.registry import GeneratorRegistry


def configure_list_generators_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    subparsers.add_parser(
        "list-generators",
        help="List all available procedural generators.",
    )


def run_list_generators_command(registry: GeneratorRegistry) -> int:
    print("Available generators:")
    for name in registry.names():
        print(f"- {name}")
    return 0
