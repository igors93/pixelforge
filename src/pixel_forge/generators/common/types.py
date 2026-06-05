"""Shared NumPy array types used by procedural generators."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
UInt8Array: TypeAlias = NDArray[np.uint8]
