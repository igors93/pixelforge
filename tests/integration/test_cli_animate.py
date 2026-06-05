"""Integration tests for the `pixelforge animate` CLI command.

These tests invoke the full pipeline end-to-end: CLI parsing → AnimationService
→ GIF encoding → file writing → manifest writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pixel_forge.animation.frame_timing import frame_duration_ms_from_fps
from pixel_forge.cli.main import main


def _animate(
    tmp_path: Path,
    *,
    generator: str = "harmonic-waves",
    seed: int = 42,
    width: int = 48,
    height: int = 48,
    frames: int = 4,
    fps: int = 12,
    profile: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Run the animate command and return the output GIF path."""
    output = tmp_path / f"{generator}.gif"
    args = [
        "animate",
        "--generator", generator,
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--frames", str(frames),
        "--fps", str(fps),
        "--output", str(output),
        "--overwrite",
    ]
    if profile:
        args += ["--motion-profile", profile]
    if extra_args:
        args += extra_args
    rc = main(args)
    assert rc == 0, f"animate command returned exit code {rc}"
    return output


# ── Basic generation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("generator", ["harmonic-waves", "radial-bloom", "plasma-flow"])
def test_animate_produces_gif(tmp_path: Path, generator: str) -> None:
    output = _animate(tmp_path, generator=generator)
    assert output.exists()
    assert output.stat().st_size > 0


def test_animate_mandelbrot(tmp_path: Path) -> None:
    output = _animate(tmp_path, generator="mandelbrot-dream", width=32, height=32, frames=4)
    assert output.exists()


# ── GIF integrity ─────────────────────────────────────────────────────────────

def test_gif_is_valid_image(tmp_path: Path) -> None:
    output = _animate(tmp_path)
    img = Image.open(output)
    assert img.format == "GIF"


def test_gif_frame_count(tmp_path: Path) -> None:
    n = 6
    output = _animate(tmp_path, frames=n)
    img = Image.open(output)
    count = 0
    try:
        while True:
            img.seek(count)
            count += 1
    except EOFError:
        pass
    assert count == n


def test_gif_dimensions(tmp_path: Path) -> None:
    output = _animate(tmp_path, width=64, height=48)
    img = Image.open(output)
    assert img.size == (64, 48)


def test_gif_loop_count_infinite(tmp_path: Path) -> None:
    output = _animate(tmp_path, frames=4)
    img = Image.open(output)
    assert img.info.get("loop", 0) == 0


def test_gif_duration_matches_fps(tmp_path: Path) -> None:
    fps = 12
    output = _animate(tmp_path, fps=fps)

    img = Image.open(output)

    expected_ms = frame_duration_ms_from_fps(fps)

    assert img.info.get("duration") == expected_ms


# ── Metadata (JSON manifest) ──────────────────────────────────────────────────

def test_manifest_created(tmp_path: Path) -> None:
    output = _animate(tmp_path)
    manifest_path = output.with_suffix(".json")
    assert manifest_path.exists()


def test_manifest_is_valid_json(tmp_path: Path) -> None:
    output = _animate(tmp_path)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert isinstance(manifest, dict)


def test_manifest_contains_key_fields(tmp_path: Path) -> None:
    output = _animate(tmp_path, seed=77)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["generator"] == "harmonic-waves"
    assert manifest["master_seed"] == 77
    assert manifest["frame_count"] == 4
    assert "motion_profile" in manifest
    assert "content_id" in manifest
    assert "temporal_quality" in manifest


def test_no_metadata_flag(tmp_path: Path) -> None:
    output = _animate(tmp_path, extra_args=["--no-metadata"])
    assert output.exists()
    assert not output.with_suffix(".json").exists()


# ── Determinism ───────────────────────────────────────────────────────────────

def test_same_seed_same_content_id(tmp_path: Path) -> None:
    out1 = (tmp_path / "a.gif")
    out2 = (tmp_path / "b.gif")
    for out in [out1, out2]:
        args = [
            "animate", "--generator", "harmonic-waves",
            "--width", "32", "--height", "32",
            "--seed", "42", "--frames", "4", "--fps", "12",
            "--output", str(out), "--overwrite",
        ]
        assert main(args) == 0

    m1 = json.loads(out1.with_suffix(".json").read_text())
    m2 = json.loads(out2.with_suffix(".json").read_text())
    assert m1["content_id"] == m2["content_id"]


# ── Motion profiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("profile", ["phase-drift", "rotating-field", "color-orbit"])
def test_harmonic_motion_profiles(tmp_path: Path, profile: str) -> None:
    output = _animate(tmp_path, generator="harmonic-waves", profile=profile)
    assert output.exists()
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["motion_profile"] == profile


@pytest.mark.parametrize("profile", ["bloom-pulse", "radial-rotation"])
def test_radial_motion_profiles(tmp_path: Path, profile: str) -> None:
    output = _animate(tmp_path, generator="radial-bloom", profile=profile)
    assert output.exists()


# ── Validation errors ─────────────────────────────────────────────────────────

def test_invalid_generator_exits_2(tmp_path: Path) -> None:
    rc = main([
        "animate", "--generator", "nonexistent-generator",
        "--output", str(tmp_path / "x.gif"),
        "--frames", "4", "--fps", "12", "--seed", "1",
    ])
    assert rc == 2


def test_invalid_frames_exits_2(tmp_path: Path) -> None:
    rc = main([
        "animate", "--generator", "harmonic-waves",
        "--output", str(tmp_path / "x.gif"),
        "--frames", "1", "--fps", "12", "--seed", "1",
    ])
    assert rc == 2


def test_invalid_motion_profile_exits_2(tmp_path: Path) -> None:
    rc = main([
        "animate", "--generator", "harmonic-waves",
        "--output", str(tmp_path / "x.gif"),
        "--frames", "4", "--fps", "12", "--seed", "1",
        "--motion-profile", "totally-bogus",
    ])
    assert rc == 2


def test_output_not_gif_exits_2(tmp_path: Path) -> None:
    rc = main([
        "animate", "--generator", "harmonic-waves",
        "--output", str(tmp_path / "x.png"),
        "--frames", "4", "--fps", "12", "--seed", "1",
    ])
    assert rc == 2


def test_overwrite_protection(tmp_path: Path) -> None:
    output = _animate(tmp_path)
    assert output.exists()
    # Try again without --overwrite
    rc = main([
        "animate", "--generator", "harmonic-waves",
        "--output", str(output),
        "--frames", "4", "--fps", "12", "--seed", "42",
    ])
    assert rc == 2


# ── Custom options ────────────────────────────────────────────────────────────

def test_custom_gif_colors(tmp_path: Path) -> None:
    output = _animate(tmp_path, extra_args=["--colors", "128"])
    assert output.exists()
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["gif_colors"] == 128


def test_floyd_steinberg_dither(tmp_path: Path) -> None:
    output = _animate(tmp_path, extra_args=["--dither", "floyd-steinberg"])
    assert output.exists()
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["gif_dither"] == "floyd-steinberg"


def test_finite_loop_count(tmp_path: Path) -> None:
    output = _animate(tmp_path, extra_args=["--loop-count", "3"])
    assert output.exists()
    img = Image.open(output)
    assert img.info.get("loop") == 3


# ── Static PNG regression (existing generate command unaffected) ──────────────

def test_generate_still_works_after_animate_import(tmp_path: Path) -> None:
    png = tmp_path / "test.png"
    rc = main([
        "generate",
        "--generator", "harmonic-waves",
        "--seed", "42",
        "--width", "32", "--height", "32",
        "--output", str(png),
        "--overwrite",
    ])
    assert rc == 0
    assert png.exists()
    assert png.stat().st_size > 0
