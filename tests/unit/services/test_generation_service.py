from pathlib import Path

from pixel_forge.core.config import Settings
from pixel_forge.core.models import GenerationRequest, ImageSize
from pixel_forge.generators.registry import build_default_registry
from pixel_forge.image.encoders import PngEncoder
from pixel_forge.image.writers import AtomicFileWriter
from pixel_forge.services import GenerationService
from pixel_forge.shared.validation import RequestValidator


def test_service_generates_a_valid_png(tmp_path: Path) -> None:
    settings = Settings()
    service = GenerationService(
        registry=build_default_registry(),
        validator=RequestValidator(settings),
        encoder=PngEncoder(compress_level=settings.png_compress_level),
        writer=AtomicFileWriter(),
        supported_output_suffixes=settings.supported_output_suffixes,
    )
    output_path = tmp_path / "generated.png"

    result = service.generate(
        GenerationRequest(
            size=ImageSize(width=16, height=12),
            generator_name="harmonic-waves",
            output_path=output_path,
            seed=42,
        )
    )

    assert result.output_path == output_path.resolve()
    assert result.seed == 42
    assert result.bytes_written > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
