"""Protocols that define replaceable PixelForge components."""

from pixel_forge.core.protocols.binary_writer import BinaryWriter
from pixel_forge.core.protocols.image_encoder import ImageEncoder
from pixel_forge.core.protocols.image_generator import ImageGenerator

__all__ = ["BinaryWriter", "ImageEncoder", "ImageGenerator"]
