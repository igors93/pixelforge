"""Deterministic animation seed and stream derivation.

Animation randomness is fully isolated from static generation randomness.
The animation seed is derived from the static candidate_seed, the generator
name, and the animation schema version via SHA-256 so that:

  * Changing any animation parameter does not affect the static recipe.
  * The same (candidate_seed, generator, schema_version) always produces the
    same animation seed and therefore the same animation recipe.
  * Animation streams use the same SeedSequence splitting strategy as static
    streams so that each sampling domain remains independent.

Static RandomStreams (traits, geometry, composition, palette, lighting,
accents, rarity, quality_retry) are never touched here.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


# Stable spawn-order names for animation-specific streams.
# Append only — never reorder — to preserve existing animation seeds.
_ANIMATION_STREAM_NAMES: tuple[str, ...] = (
    "anim_motion",       # 0 – motion profile parameters (turns, cycles, etc.)
    "anim_color",        # 1 – color cycle and palette phase animation
    "anim_camera",       # 2 – camera / zoom / center movement
    "anim_accents",      # 3 – animated accent and rare-event animation
    "anim_rarity",       # 4 – animation rarity sampling
    "anim_retry",        # 5 – animation retry candidate derivation
)


def derive_animation_seed(
    *,
    candidate_seed: int,
    generator_name: str,
    animation_schema_version: str,
) -> int:
    """Return a deterministic 64-bit seed for animation randomness.

    Completely independent of static seed derivation (uses the prefix
    "animation" to guarantee a different hash even for equal inputs).
    """
    payload = "|".join([
        "animation",
        str(candidate_seed),
        generator_name,
        animation_schema_version,
    ]).encode()
    digest = hashlib.sha256(payload).digest()
    value: int = struct.unpack_from(">Q", digest)[0]
    return value


def derive_animation_retry_seed(
    *,
    animation_seed: int,
    retry_index: int,
) -> int:
    """Deterministic seed for an animation retry attempt."""
    payload = "|".join(["anim_retry", str(animation_seed), str(retry_index)]).encode()
    digest = hashlib.sha256(payload).digest()
    value: int = struct.unpack_from(">Q", digest)[0]
    return value


class AnimationStreams:
    """Independent NumPy generators for each animation sampling domain."""

    __slots__ = (
        "motion",
        "color",
        "camera",
        "accents",
        "rarity",
        "retry",
    )

    def __init__(
        self,
        motion: np.random.Generator,
        color: np.random.Generator,
        camera: np.random.Generator,
        accents: np.random.Generator,
        rarity: np.random.Generator,
        retry: np.random.Generator,
    ) -> None:
        self.motion = motion
        self.color = color
        self.camera = camera
        self.accents = accents
        self.rarity = rarity
        self.retry = retry

    @classmethod
    def from_seed(cls, seed: int) -> AnimationStreams:
        """Create independent streams from a single animation seed."""
        sequence = np.random.SeedSequence(seed)
        children = sequence.spawn(len(_ANIMATION_STREAM_NAMES))
        rngs = [np.random.default_rng(child) for child in children]
        return cls(
            motion=rngs[0],
            color=rngs[1],
            camera=rngs[2],
            accents=rngs[3],
            rarity=rngs[4],
            retry=rngs[5],
        )
