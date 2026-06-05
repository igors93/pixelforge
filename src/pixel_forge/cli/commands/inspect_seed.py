"""Implementation of the ``inspect-seed`` CLI command.

Prints the recipe, rarity tier, trait probabilities, and compatibility rule
changes for a given seed without writing an image unless --save is requested.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys

from pixel_forge.aesthetics.compatibility.recipe_compatibility_validator import (
    RecipeCompatibilityValidator,
)
from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationOptions, GenerationRequest, ImageSize
from pixel_forge.generators.common.recipe_generator import RecipeGenerator
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.metadata.artwork_manifest import _recipe_to_dict
from pixel_forge.randomness.deterministic_retry import derive_candidate_seed
from pixel_forge.randomness.random_streams import RandomStreams
from pixel_forge.rarity.rarity_evaluator import RarityEvaluator


def configure_inspect_seed_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    settings: Settings,
) -> None:
    parser = subparsers.add_parser(
        "inspect-seed",
        help="Print the recipe, rarity, and compatibility changes for a seed without rendering.",
    )
    parser.add_argument(
        "--generator",
        default=settings.default_generator,
        help=f"Generator name (default: {settings.default_generator}).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Master seed.")
    parser.add_argument(
        "--width", type=int, default=settings.default_width, help="Canvas width."
    )
    parser.add_argument(
        "--height", type=int, default=settings.default_height, help="Canvas height."
    )
    parser.add_argument(
        "--palette", default=None, metavar="PALETTE_NAME", help="Force a named palette."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text.",
    )


def run_inspect_seed_command(
    arguments: argparse.Namespace,
    registry: GeneratorRegistry,
) -> int:
    generator = registry.get(arguments.generator)
    if not isinstance(generator, RecipeGenerator):
        print(
            f"Error: generator '{arguments.generator}' does not support recipe inspection.",
            file=sys.stderr,
        )
        return 2

    master_seed = arguments.seed if arguments.seed is not None else secrets.randbits(64)
    candidate_seed = derive_candidate_seed(
        master_seed=master_seed,
        generator_name=generator.name,
        retry_index=0,
        schema_version="1.0",
    )

    options = GenerationOptions(palette_name=arguments.palette)
    request = GenerationRequest(
        size=ImageSize(width=arguments.width, height=arguments.height),
        generator_name=generator.name,
        output_path=__import__("pathlib").Path("inspect-output.png"),
        seed=master_seed,
        options=options,
    )

    streams = RandomStreams.from_seed(candidate_seed)
    recipe, trait_probs = generator.build_recipe(
        request, streams, candidate_seed=candidate_seed, retry_index=0
    )

    compat_validator = RecipeCompatibilityValidator()
    compat_result = compat_validator.validate(recipe)
    recipe = compat_result.recipe

    rarity_evaluator = RarityEvaluator()
    rarity = rarity_evaluator.evaluate(trait_probs)

    if arguments.json:
        output = {
            "master_seed": master_seed,
            "candidate_seed": candidate_seed,
            "recipe": _recipe_to_dict(recipe),
            "rarity": {
                "tier": rarity.overall_tier,
                "information_bits": rarity.total_information_bits,
                "summary": rarity.summary,
                "traits": {
                    name: {
                        "value": entry.value,
                        "probability": entry.probability,
                        "information_bits": entry.information_bits,
                    }
                    for name, entry in rarity.trait_details.items()
                },
            },
            "applied_compatibility_rules": list(compat_result.applied_rules),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Generator         : {recipe.generator_name}")
        print(f"Master seed       : {master_seed}")
        print(f"Candidate seed    : {candidate_seed}")
        print(f"Canvas            : {recipe.width}x{recipe.height}")
        print(f"Palette           : {recipe.palette_name}")
        print(f"Symmetry          : {recipe.symmetry_mode.value}")
        print(f"Complexity        : {recipe.complexity_level.value}")
        print(f"Detail            : {recipe.detail_level.value}")
        print(f"Background        : {recipe.background_mode.value}")
        print(f"Lighting          : {recipe.lighting_mode.value}")
        print(f"Accent            : {recipe.accent_mode.value}")
        if recipe.rare_events:
            print(f"Rare events       : {', '.join(recipe.rare_events)}")
        else:
            print("Rare events       : (none)")
        if compat_result.applied_rules:
            print(f"Compat. rules     : {', '.join(compat_result.applied_rules)}")
        else:
            print("Compat. rules     : (none applied)")
        print()
        print(f"Rarity tier       : {rarity.overall_tier}")
        print(f"Information bits  : {rarity.total_information_bits:.2f}")
        print(f"Summary           : {rarity.summary}")
        print()
        if rarity.most_significant_traits:
            print("Most significant traits:")
            for entry in rarity.most_significant_traits:
                print(
                    f"  {entry.trait_name:24s} {entry.value:20s} "
                    f"p={entry.probability*100:.2f}%  "
                    f"bits={entry.information_bits:.2f}"
                )
        print()
        print("Generator params:")
        for key, value in recipe.generator_params.items():
            if isinstance(value, float):
                print(f"  {key:28s} {value:.6f}")
            else:
                print(f"  {key:28s} {value}")

    return 0
