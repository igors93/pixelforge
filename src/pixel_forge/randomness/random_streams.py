"""Independent deterministic random streams derived from a single master seed.

Splitting the master seed into named child streams prevents a change in one
sampling domain (e.g. palette) from altering another (e.g. geometry) for the
same master seed. Without independent streams, every call to the single shared
generator shifts the internal state and breaks long-term deterministic stability
whenever any sampling step is added, removed, or reordered.

numpy.random.SeedSequence provides cryptographically seeded, reproducible child
sequences. Its spawning algorithm is stable across NumPy versions and does not
depend on Python's built-in hash(), which is not stable across processes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stable spawn-order indices for each named stream.
# Inserting a new name requires appending to the end – never reordering –
# to preserve the meaning of existing seeds.
_STREAM_NAMES: tuple[str, ...] = (
    "traits",       # 0 – primary trait sampling (petals, symmetry, complexity …)
    "geometry",     # 1 – spatial parameters (frequencies, rotations, offsets …)
    "composition",  # 2 – layer counts, blend modes, phase relationships
    "palette",      # 3 – palette selection and color phase shift
    "lighting",     # 4 – brightness, highlight, shadow parameters
    "accents",      # 5 – rare-event and accent sampling
    "rarity",       # 6 – rarity tier draw and event probability sampling
    "quality_retry",# 7 – deterministic retry candidate seed derivation
)


@dataclass(frozen=True, slots=True)
class RandomStreams:
    """One independent NumPy generator per named sampling domain.

    Each generator is derived from the master seed via SeedSequence.spawn so
    that sampling in one domain never affects the state of another domain.
    """

    traits: np.random.Generator
    geometry: np.random.Generator
    composition: np.random.Generator
    palette: np.random.Generator
    lighting: np.random.Generator
    accents: np.random.Generator
    rarity: np.random.Generator
    quality_retry: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> RandomStreams:
        """Create independent streams from a single master integer seed."""
        sequence = np.random.SeedSequence(seed)
        children = sequence.spawn(len(_STREAM_NAMES))
        rngs = [np.random.default_rng(child) for child in children]
        return cls(
            traits=rngs[0],
            geometry=rngs[1],
            composition=rngs[2],
            palette=rngs[3],
            lighting=rngs[4],
            accents=rngs[5],
            rarity=rngs[6],
            quality_retry=rngs[7],
        )
