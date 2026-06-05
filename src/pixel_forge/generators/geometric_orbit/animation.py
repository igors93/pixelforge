"""Reserved extension point for future geometric-orbit GIF support."""

from __future__ import annotations


class GeometricOrbitAnimationNotImplementedError(NotImplementedError):
    """Raised when animation support is requested before implementation."""


def build_animation_support() -> None:
    """Signal that this update improves static rendering only."""

    raise GeometricOrbitAnimationNotImplementedError(
        "Animation support for 'geometric-orbit' is not included in this update."
    )
