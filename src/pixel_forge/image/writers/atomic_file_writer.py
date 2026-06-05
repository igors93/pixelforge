"""Safe local filesystem writer."""

import os
import tempfile
from pathlib import Path

from pixel_forge.core.exceptions import OutputFileExistsError, OutputWriteError


class AtomicFileWriter:
    """Write a complete file atomically to avoid partially written images."""

    def write(self, data: bytes, output_path: Path, *, overwrite: bool) -> Path:
        normalized_path = output_path.expanduser().resolve(strict=False)
        parent = normalized_path.parent

        if normalized_path.exists() and not overwrite:
            raise OutputFileExistsError(
                f"Output file already exists: {normalized_path}. Use --overwrite to replace it."
            )

        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OutputWriteError(
                f"Could not create output directory '{parent}': {error}."
            ) from error

        temporary_path: Path | None = None
        try:
            # The temporary file is created in the destination directory so
            # os.replace remains atomic on the same filesystem.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{normalized_path.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, normalized_path)
            return normalized_path
        except OSError as error:
            raise OutputWriteError(
                f"Could not write output file '{normalized_path}': {error}."
            ) from error
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
