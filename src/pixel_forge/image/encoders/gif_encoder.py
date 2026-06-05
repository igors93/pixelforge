"""GIF encoder with deterministic global palette.

Design decisions:
  - Global palette is built from 4 representative frames (phases 0.0, 0.25,
    0.5, 0.75). Using representative frames avoids per-frame palette drift
    (colour flicker) while keeping quantisation cost O(4 × frame_pixels).
  - All frames are quantised against the same global palette image, which
    Pillow applies via ``Image.quantize(palette=palette_img, dither=...)``.
  - Frames are kept as palette-indexed (mode "P") PIL Images. Full RGB arrays
    are released after quantisation to keep peak memory proportional to
    O(representative_frames × pixels + total_frames × pixels/3).
  - The GIF is assembled in one ``save(..., save_all=True, append_images=...)``
    call using ``disposal=2`` (restore to background) between frames.
  - Duration is derived as ``round(1000 / fps)`` milliseconds per frame.

Limitations:
  - GIF colour depth is limited to 8 bits (256 colours). High-frequency
    gradients will show banding regardless of palette size.
  - Floyd-Steinberg dithering improves quality at the cost of slightly larger
    file sizes and minor colour inconsistencies across frames.
"""

from __future__ import annotations

import io
from typing import Iterator

import numpy as np
from PIL import Image

from pixel_forge.generators.common.types import UInt8Array
from pixel_forge.image.encoders.gif_encoding_options import GifEncodingOptions

# Maximum palette entries Pillow will accept for GIF.
_MAX_GIF_COLORS = 256


class GifEncoder:
    """Encode a sequence of RGB uint8 frames into GIF bytes."""

    def __init__(self, options: GifEncodingOptions | None = None) -> None:
        self._opts = options or GifEncodingOptions()

    def encode(
        self,
        frames: list[UInt8Array],
        *,
        frame_duration_ms: int,
    ) -> bytes:
        """Encode *frames* to GIF and return the raw bytes.

        All frames must have the same (height, width, 3) shape and dtype uint8.
        ``frame_duration_ms`` is the display time for every frame in milliseconds.
        """
        if len(frames) < 2:
            raise ValueError(f"GIF requires at least 2 frames, got {len(frames)}")

        gif_colors = min(max(self._opts.gif_colors, 2), _MAX_GIF_COLORS)
        dither_mode = (
            Image.Dither.FLOYDSTEINBERG
            if self._opts.dither == "floyd-steinberg"
            else Image.Dither.NONE
        )

        # Build global palette from representative frames.
        palette_img = self._build_global_palette(frames, gif_colors)

        # Quantise all frames against the shared palette.
        pil_frames = [
            self._quantise_frame(f, palette_img, dither_mode)
            for f in frames
        ]

        # Encode to GIF bytes.
        buf = io.BytesIO()
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=self._opts.loop_count,
            disposal=2,
            optimize=False,
        )
        return buf.getvalue()

    # ──────────────────────────────────────────────────────────────────────────

    def _build_global_palette(
        self,
        frames: list[UInt8Array],
        gif_colors: int,
    ) -> Image.Image:
        """Build a quantised palette image from representative frames.

        Uses frames at indices 0, N//4, N//2, 3N//4 as representative samples.
        These four samples capture the full range of a single-cycle periodic
        animation without rendering additional frames.
        """
        n = len(frames)
        rep_indices = sorted({0, n // 4, n // 2, 3 * n // 4})
        rep_frames = [frames[i] for i in rep_indices]

        # Stack representative frames into a tall strip for quantisation.
        h, w = frames[0].shape[:2]
        stacked = np.vstack(rep_frames)                 # shape (reps*h, w, 3)
        strip_img = Image.fromarray(stacked.astype(np.uint8), "RGB")

        # Quantise the strip to the desired palette size.
        # MEDIANCUT gives the most stable palette across many PixelForge images.
        palette_img = strip_img.quantize(
            colors=gif_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        return palette_img

    @staticmethod
    def _quantise_frame(
        frame: UInt8Array,
        palette_img: Image.Image,
        dither_mode: Image.Dither,
    ) -> Image.Image:
        """Apply the global palette to one frame and return a mode-P PIL image."""
        pil = Image.fromarray(frame.astype(np.uint8), "RGB")
        return pil.quantize(palette=palette_img, dither=int(dither_mode))
