from pathlib import Path

from PIL import Image

from pixel_forge.cli.main import main


def test_generate_command_creates_requested_image(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-output.png"

    exit_code = main(
        [
            "generate",
            "--width",
            "20",
            "--height",
            "10",
            "--seed",
            "99",
            "--output",
            str(output_path),
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
            "--width",
            "1001",
            "--height",
            "10",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()
