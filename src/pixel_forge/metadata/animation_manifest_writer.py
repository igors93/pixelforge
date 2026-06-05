"""Atomic animation manifest writer.

Writes the JSON manifest alongside the GIF using the same temp-file-and-rename
strategy as the static ManifestWriter so that the two files are never in an
inconsistent state.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pixel_forge.metadata.animation_manifest import AnimationManifest, manifest_to_json


class AnimationManifestWriter:
    """Write an AnimationManifest atomically to disk."""

    def write(
        self,
        manifest: AnimationManifest,
        gif_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write the manifest as <gif_stem>.json next to the GIF file.

        Returns the path of the written JSON file. Raises FileExistsError if
        the file already exists and *overwrite* is False.
        """
        json_path = gif_path.with_suffix(".json")
        if json_path.exists() and not overwrite:
            from pixel_forge.core.exceptions import OutputFileExistsError
            raise OutputFileExistsError(
                f"Manifest already exists: {json_path}. Use --overwrite to replace."
            )

        content = manifest_to_json(manifest).encode("utf-8")
        dir_path = json_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, json_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return json_path
