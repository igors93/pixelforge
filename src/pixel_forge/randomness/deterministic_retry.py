"""Deterministic candidate seed derivation for quality retries.

When a generated image fails minimum quality thresholds, the system retries
with a different candidate seed. That seed must be derived deterministically
from the master seed, generator name, retry index, and schema version so that
the same inputs always produce the same retry sequence – preserving the
guarantee that (master_seed, generator, dimensions) → (recipe, pixels) is
stable from one version of PixelForge to the next.

Python's built-in hash() is intentionally excluded because it is not stable
across interpreter invocations (PYTHONHASHSEED randomization).
"""

from __future__ import annotations

import hashlib
import struct


def derive_candidate_seed(
    *,
    master_seed: int,
    generator_name: str,
    retry_index: int,
    schema_version: str,
) -> int:
    """Return a deterministic 64-bit candidate seed for the given retry attempt.

    Retry index 0 is the first (unretried) attempt; index 1 is the first
    actual retry, and so on. The master seed is never modified.
    """
    payload = "|".join(
        [
            str(master_seed),
            generator_name,
            str(retry_index),
            schema_version,
        ]
    ).encode()

    digest = hashlib.sha256(payload).digest()
    # Read the first 8 bytes as an unsigned 64-bit integer (big-endian).
    value: int = struct.unpack_from(">Q", digest)[0]
    return value
