"""Application service that coordinates image generation."""

from pixel_forge.core.models import GenerationRequest, GenerationResult
from pixel_forge.core.protocols import BinaryWriter, ImageEncoder
from pixel_forge.generators.registry import GeneratorRegistry
from pixel_forge.shared.paths import normalize_output_path
from pixel_forge.shared.validation import RequestValidator


class GenerationService:
    """Execute one complete, validated image generation operation."""

    def __init__(
        self,
        *,
        registry: GeneratorRegistry,
        validator: RequestValidator,
        encoder: ImageEncoder,
        writer: BinaryWriter,
        supported_output_suffixes: tuple[str, ...],
    ) -> None:
        self._registry = registry
        self._validator = validator
        self._encoder = encoder
        self._writer = writer
        self._supported_output_suffixes = supported_output_suffixes

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._validator.validate(request)
        output_path = normalize_output_path(
            request.output_path,
            supported_suffixes=self._supported_output_suffixes,
        )

        generator = self._registry.get(request.generator_name)
        image = generator.generate(request)
        encoded_image = self._encoder.encode(image)
        final_path = self._writer.write(
            encoded_image,
            output_path,
            overwrite=request.overwrite,
        )

        return GenerationResult(
            output_path=final_path,
            size=image.size,
            generator_name=image.generator_name,
            seed=image.seed,
            bytes_written=len(encoded_image),
        )
