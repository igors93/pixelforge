"""Contract implemented by binary storage writers."""

from pathlib import Path
from typing import Protocol


class BinaryWriter(Protocol):
    """Persist binary data and return the final normalized path."""

    def write(self, data: bytes, output_path: Path, *, overwrite: bool) -> Path:
        """Write bytes to storage."""

        ...
