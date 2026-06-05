"""Tests for the GIF encoder."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from pixel_forge.image.encoders.gif_encoder import GifEncoder
from pixel_forge.image.encoders.gif_encoding_options import GifEncodingOptions


def _gradient_frame(offset: int, h: int = 32, w: int = 32) -> np.ndarray:
    """Create a simple gradient frame for testing."""
    x = np.linspace(offset, offset + 200, w, dtype=np.float32)
    col = np.clip(x[np.newaxis, :, np.newaxis], 0, 255)
    return np.broadcast_to(np.repeat(col, 3, axis=2), (h, w, 3)).astype(np.uint8).copy()


def test_encoder_produces_bytes() -> None:
    encoder = GifEncoder()
    frames = [_gradient_frame(i * 20) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_encoder_requires_at_least_2_frames() -> None:
    encoder = GifEncoder()
    with pytest.raises(ValueError, match="at least 2"):
        encoder.encode([_gradient_frame(0)], frame_duration_ms=100)


def test_encoded_gif_is_valid() -> None:
    encoder = GifEncoder()
    frames = [_gradient_frame(i * 30) for i in range(6)]
    data = encoder.encode(frames, frame_duration_ms=83)
    img = Image.open(io.BytesIO(data))
    assert img.format == "GIF"


def test_encoded_gif_frame_count() -> None:
    """Pillow reports correct number of frames."""
    n = 8
    encoder = GifEncoder()
    frames = [_gradient_frame(i * 25) for i in range(n)]
    data = encoder.encode(frames, frame_duration_ms=41)
    img = Image.open(io.BytesIO(data))
    # Count frames via seek
    count = 0
    try:
        while True:
            img.seek(count)
            count += 1
    except EOFError:
        pass
    assert count == n


def test_encoded_gif_dimensions() -> None:
    h, w = 48, 64
    encoder = GifEncoder()
    frames = [_gradient_frame(i * 20, h=h, w=w) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    img = Image.open(io.BytesIO(data))
    assert img.size == (w, h)


def test_encoded_gif_loop_count_infinite() -> None:
    opts = GifEncodingOptions(loop_count=0)
    encoder = GifEncoder(opts)
    frames = [_gradient_frame(i * 30) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    img = Image.open(io.BytesIO(data))
    # Loop count 0 means infinite; Pillow stores it as 0 in the NETSCAPE block.
    assert img.info.get("loop", 0) == 0


def test_encoded_gif_duration() -> None:
    duration_ms = 83
    encoder = GifEncoder()
    frames = [_gradient_frame(i * 30) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=duration_ms)
    img = Image.open(io.BytesIO(data))
    assert img.info.get("duration") == duration_ms


def test_gif_colors_option_256() -> None:
    opts = GifEncodingOptions(gif_colors=256)
    encoder = GifEncoder(opts)
    frames = [_gradient_frame(i * 20) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    assert len(data) > 0


def test_gif_colors_option_64() -> None:
    opts = GifEncodingOptions(gif_colors=64)
    encoder = GifEncoder(opts)
    frames = [_gradient_frame(i * 20) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    assert len(data) > 0


def test_gif_dithering_floyd_steinberg() -> None:
    opts = GifEncodingOptions(dither="floyd-steinberg")
    encoder = GifEncoder(opts)
    frames = [_gradient_frame(i * 20) for i in range(4)]
    data = encoder.encode(frames, frame_duration_ms=100)
    assert len(data) > 0
