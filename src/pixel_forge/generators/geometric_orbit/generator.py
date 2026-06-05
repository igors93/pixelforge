"""Geometric-orbit procedural image generator."""

from __future__ import annotations

from numpy.random import Generator

from pixel_forge.core.models import GenerationRequest
from pixel_forge.generators.common import SeededArrayGenerator, UInt8Array
from pixel_forge.generators.geometric_orbit.recipe_builder import build_recipe
from pixel_forge.generators.geometric_orbit.renderer import GeometricOrbitRenderer


class GeometricOrbitGenerator(SeededArrayGenerator):
    """Generate clean poster-like geometric compositions from a seed."""

    def __init__(self) -> None:
        self._renderer = GeometricOrbitRenderer()

    @property
    def name(self) -> str:
        return "geometric-orbit"

    def render(
        self,
        request: GenerationRequest,
        random_source: Generator,
    ) -> UInt8Array:
        recipe = build_recipe(random_source)
        return self._renderer.render(size=request.size, recipe=recipe)
