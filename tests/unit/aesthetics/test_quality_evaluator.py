"""Tests for the quality evaluator and metrics."""

from __future__ import annotations

import numpy as np

from pixel_forge.aesthetics.quality.quality_evaluator import QualityEvaluator, QualityThresholds


def _solid_rgb(r: int, g: int, b: int, size: int = 64) -> np.ndarray[np.uint8]:
    """Create a solid-color image for edge-case testing."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    return arr


def _gradient_rgb(size: int = 64) -> np.ndarray[np.uint8]:
    """Create a gradient image with good contrast."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        arr[i, :, 0] = int(i * 255 / size)
        arr[i, :, 1] = int(i * 128 / size)
        arr[:, i, 2] = int(i * 200 / size)
    return arr


def test_pure_black_is_rejected() -> None:
    evaluator = QualityEvaluator()
    rgb = _solid_rgb(0, 0, 0)
    result = evaluator.evaluate(rgb)
    assert not result.accepted
    assert len(result.rejection_reasons) > 0


def test_pure_white_is_rejected() -> None:
    evaluator = QualityEvaluator()
    rgb = _solid_rgb(255, 255, 255)
    result = evaluator.evaluate(rgb)
    assert not result.accepted


def test_gradient_image_is_accepted() -> None:
    evaluator = QualityEvaluator(QualityThresholds(min_aggregate_score=0.10))
    rgb = _gradient_rgb(64)
    result = evaluator.evaluate(rgb)
    assert result.accepted


def test_aggregate_score_in_range() -> None:
    evaluator = QualityEvaluator()
    rgb = _gradient_rgb(64)
    result = evaluator.evaluate(rgb)
    assert 0.0 <= result.aggregate_score <= 1.0


def test_quality_result_metrics_are_in_range() -> None:
    evaluator = QualityEvaluator()
    rgb = _gradient_rgb(64)
    result = evaluator.evaluate(rgb)
    assert 0.0 <= result.luminance_contrast <= 1.0
    assert 0.0 <= result.clipped_black_ratio <= 1.0
    assert 0.0 <= result.clipped_white_ratio <= 1.0
    assert 0.0 <= result.mean_saturation <= 1.0
    assert 0.0 <= result.color_diversity <= 1.0
    assert 0.0 <= result.visual_entropy <= 1.0
    assert 0.0 <= result.edge_density <= 1.0
    assert 0.0 <= result.horizontal_symmetry <= 1.0
    assert 0.0 <= result.vertical_symmetry <= 1.0


def test_custom_threshold_applies() -> None:
    evaluator = QualityEvaluator(QualityThresholds(min_aggregate_score=0.99))
    rgb = _gradient_rgb(64)
    result = evaluator.evaluate(rgb)
    # A gradient with threshold 0.99 is almost certainly rejected.
    assert not result.accepted


def test_tiny_image_does_not_crash() -> None:
    evaluator = QualityEvaluator()
    rgb = _solid_rgb(128, 64, 32, size=2)
    result = evaluator.evaluate(rgb)
    assert isinstance(result.aggregate_score, float)
