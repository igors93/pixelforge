"""Atomic JSON manifest writer."""

from __future__ import annotations

from pathlib import Path

from pixel_forge.image.writers.atomic_file_writer import AtomicFileWriter
from pixel_forge.metadata.artwork_manifest import ArtworkManifest, manifest_to_json


class ManifestWriter:
    """Write artwork manifests as atomic JSON files beside their PNG."""

    def __init__(self) -> None:
        self._writer = AtomicFileWriter()

    def write(
        self,
        manifest: ArtworkManifest,
        png_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Serialize *manifest* to JSON and write it beside *png_path*.

        The JSON file shares the PNG's stem and sits in the same directory.
        Writing is atomic: the file either appears complete or not at all.
        """
        json_path = png_path.with_suffix(".json")
        json_bytes = manifest_to_json(manifest).encode("utf-8")
        return self._writer.write(json_bytes, json_path, overwrite=overwrite)
