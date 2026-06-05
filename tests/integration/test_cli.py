"""Integration tests for all CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pixel_forge.cli.main import main

# ------------------------------------------------------------------ #
# generate command – existing behavior                                #
# ------------------------------------------------------------------ #


def test_generate_command_creates_requested_image(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-output.png"

    exit_code = main(
        [
            "generate",
            "--width", "20",
            "--height", "10",
            "--seed", "99",
            "--output", str(output_path),
        ]
    )

    assert exit_code == 0
    with Image.open(output_path) as generated_image:
        assert generated_image.format == "PNG"
        assert generated_image.mode == "RGB"
        assert generated_image.size == (20, 10)


def test_generate_command_returns_domain_error_exit_code(tmp_path: Path) -> None:
    output_path = tmp_path / "invalid.png"

    exit_code = main(
        [
            "generate",
            "--width", "1001",
            "--height", "10",
            "--output", str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


# ------------------------------------------------------------------ #
# generate command – new options                                      #
# ------------------------------------------------------------------ #


def test_generate_writes_json_metadata_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "with-meta.png"

    exit_code = main(
        [
            "generate",
            "--width", "16",
            "--height", "16",
            "--seed", "1",
            "--generator", "radial-bloom",
            "--output", str(output_path),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    json_path = output_path.with_suffix(".json")
    assert json_path.exists(), "JSON manifest should be written by default"
    data = json.loads(json_path.read_text())
    assert data["generator"] == "radial-bloom"
    assert data["master_seed"] == 1


def test_generate_no_metadata_suppresses_json(tmp_path: Path) -> None:
    output_path = tmp_path / "no-meta.png"

    exit_code = main(
        [
            "generate",
            "--width", "16",
            "--height", "16",
            "--seed", "2",
            "--output", str(output_path),
            "--no-metadata",
        ]
    )

    assert exit_code == 0
    json_path = output_path.with_suffix(".json")
    assert not json_path.exists(), "JSON manifest should be absent with --no-metadata"


def test_generate_with_explicit_palette(tmp_path: Path) -> None:
    output_path = tmp_path / "palette-test.png"

    exit_code = main(
        [
            "generate",
            "--width", "16",
            "--height", "16",
            "--seed", "3",
            "--generator", "harmonic-waves",
            "--palette", "solar-flare",
            "--output", str(output_path),
            "--no-metadata",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_generate_all_generators_produce_valid_output(tmp_path: Path) -> None:
    for generator_name in [
        "harmonic-waves",
        "plasma-flow",
        "radial-bloom",
        "mandelbrot-dream",
        "random-noise",
    ]:
        output_path = tmp_path / f"{generator_name}.png"
        exit_code = main(
            [
                "generate",
                "--width", "16",
                "--height", "16",
                "--seed", "42",
                "--generator", generator_name,
                "--output", str(output_path),
                "--no-metadata",
                "--overwrite",
            ]
        )
        assert exit_code == 0, f"Generator '{generator_name}' failed"
        assert output_path.exists()


def test_generate_rectangular_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "rect.png"

    exit_code = main(
        [
            "generate",
            "--width", "64",
            "--height", "32",
            "--seed", "77",
            "--generator", "harmonic-waves",
            "--output", str(output_path),
            "--no-metadata",
        ]
    )

    assert exit_code == 0
    with Image.open(output_path) as img:
        assert img.size == (64, 32)


def test_generate_overwrite_protection(tmp_path: Path) -> None:
    output_path = tmp_path / "overwrite-test.png"
    output_path.write_bytes(b"placeholder")

    exit_code = main(
        [
            "generate",
            "--width", "16",
            "--height", "16",
            "--seed", "5",
            "--output", str(output_path),
            "--no-metadata",
        ]
    )

    # Should fail with exit code 2 because overwrite is not specified.
    assert exit_code == 2


def test_generate_deterministic_output_for_same_seed(tmp_path: Path) -> None:
    out1 = tmp_path / "det1.png"
    out2 = tmp_path / "det2.png"

    for out in (out1, out2):
        main(
            [
                "generate",
                "--width", "32",
                "--height", "32",
                "--seed", "999",
                "--generator", "radial-bloom",
                "--output", str(out),
                "--no-metadata",
            ]
        )

    assert out1.read_bytes() == out2.read_bytes()


# ------------------------------------------------------------------ #
# inspect-seed command                                                #
# ------------------------------------------------------------------ #


def test_inspect_seed_exits_successfully(tmp_path: Path) -> None:
    exit_code = main(
        [
            "inspect-seed",
            "--generator", "radial-bloom",
            "--seed", "42",
            "--width", "64",
            "--height", "64",
        ]
    )
    assert exit_code == 0


def test_inspect_seed_json_mode(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "inspect-seed",
            "--generator", "radial-bloom",
            "--seed", "7",
            "--width", "32",
            "--height", "32",
            "--json",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "recipe" in data
    assert "rarity" in data
    assert data["master_seed"] == 7


# ------------------------------------------------------------------ #
# explore command                                                     #
# ------------------------------------------------------------------ #


def test_explore_command_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "exploration"

    exit_code = main(
        [
            "explore",
            "--generator", "radial-bloom",
            "--count", "3",
            "--keep", "2",
            "--width", "16",
            "--height", "16",
            "--output", str(output_dir),
            "--no-contact-sheet",
            "--seed", "42",
        ]
    )

    assert exit_code == 0
    assert output_dir.exists()
    # Summary JSON should be present.
    summaries = list(output_dir.glob("*exploration-summary.json"))
    assert len(summaries) == 1


def test_explore_summary_has_correct_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "explore-test"
    main(
        [
            "explore",
            "--generator", "harmonic-waves",
            "--count", "4",
            "--keep", "2",
            "--width", "16",
            "--height", "16",
            "--output", str(output_dir),
            "--no-contact-sheet",
            "--seed", "10",
        ]
    )

    summary_file = next(output_dir.glob("*exploration-summary.json"))
    data = json.loads(summary_file.read_text())
    assert data["generator"] == "harmonic-waves"
    assert data["total_generated"] == 4
    assert len(data["candidates"]) == 2
