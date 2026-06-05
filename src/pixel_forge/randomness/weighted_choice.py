"""Generic weighted sampler for deterministic trait selection.

Centralizing probability distributions in explicit WeightedChoice sequences
makes the likelihood of each visual outcome readable and testable. It also
keeps probability normalization in one location rather than scattered across
multiple generator implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WeightedChoice(Generic[T]):
    """A candidate value and its relative sampling weight."""

    value: T
    weight: float


@dataclass(frozen=True, slots=True)
class SamplingResult(Generic[T]):
    """The result of one weighted draw, including the normalized probability."""

    value: T
    probability: float  # normalized weight of the selected item (sum-to-one)


def sample_weighted(
    choices: Sequence[WeightedChoice[T]],
    rng: np.random.Generator,
) -> SamplingResult[T]:
    """Draw one item from *choices* proportionally to its weight.

    Raises
    ------
    ValueError
        If *choices* is empty, any weight is negative, or the total weight is zero.
    """
    if not choices:
        raise ValueError("Cannot sample from an empty sequence of choices.")

    weights = [c.weight for c in choices]

    if any(w < 0.0 for w in weights):
        raise ValueError("All sampling weights must be non-negative.")

    total = sum(weights)
    if total <= 0.0:
        raise ValueError("The total weight of all choices must be greater than zero.")

    probabilities = np.asarray([w / total for w in weights], dtype=np.float64)

    # rng.choice draws an index; we then read the corresponding choice.
    index = int(rng.choice(len(choices), p=probabilities))
    return SamplingResult(value=choices[index].value, probability=float(probabilities[index]))
