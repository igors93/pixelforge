"""Tests for temporal quality evaluator and individual metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_forge.animation.quality import TemporalQualityEvaluator, TemporalQualityThresholds
from pixel_forge.animation.quality.temporal_quality_metrics import (
    compute_flicker_score,
    compute_luminance_variation,
    compute_motion_energy,
    compute_seam_score,
)


def _solid_frame(value: int, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_frame(offset: int, h: int = 32, w: int = 32) -> np.ndarray:
    x = np.linspace(offset, offset + 200, w, dtype=np.float32)
    frame = np.broadcast_to(x[np.newaxis, :, np.newaxis], (h, w, 1))
    rgb = np.clip(np.repeat(frame, 3, axis=2), 0, 255).astype(np.uint8)
    return rgb


# ── Seam score ────────────────────────────────────────────────────────────────

def test_seam_score_identical_frames() -> None:
    f = _solid_frame(128)
    score = compute_seam_score(f, f)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_seam_score_totally_different() -> None:
    f0 = _solid_frame(0)
    fn = _solid_frame(255)
    score = compute_seam_score(f0, fn)
    assert score < 0.1


# ── Motion energy ─────────────────────────────────────────────────────────────

def test_motion_energy_static_frames() -> None:
    frames = [_solid_frame(100)] * 8
    score = compute_motion_energy(frames)
    assert score == 0.0


def test_motion_energy_moving_frames() -> None:
    frames = [_gradient_frame(i * 20) for i in range(8)]
    score = compute_motion_energy(frames)
    assert score > 0.0


def test_motion_energy_single_frame() -> None:
    score = compute_motion_energy([_solid_frame(50)])
    assert score == 0.0


# ── Flicker score ─────────────────────────────────────────────────────────────

def test_flicker_score_consistent() -> None:
    frames = [_gradient_frame(i * 10) for i in range(8)]
    score = compute_flicker_score(frames)
    assert score > 0.5


def test_flicker_score_few_frames() -> None:
    score = compute_flicker_score([_solid_frame(10), _solid_frame(20)])
    assert score == pytest.approx(1.0)


# ── Luminance variation ───────────────────────────────────────────────────────

def test_luminance_variation_no_change() -> None:
    frames = [_solid_frame(100)] * 4
    score = compute_luminance_variation(frames)
    assert score == 0.0


def test_luminance_variation_large_change() -> None:
    frames = [_solid_frame(0), _solid_frame(200), _solid_frame(0), _solid_frame(200)]
    score = compute_luminance_variation(frames)
    assert score == pytest.approx(1.0)


# ── TemporalQualityEvaluator ──────────────────────────────────────────────────

def test_evaluator_rejects_single_frame() -> None:
    evaluator = TemporalQualityEvaluator()
    result = evaluator.evaluate([_solid_frame(100)])
    assert not result.accepted
    assert len(result.rejection_reasons) > 0


def test_evaluator_accepts_good_animation() -> None:
    evaluator = TemporalQualityEvaluator(
        TemporalQualityThresholds(
            min_aggregate_score=0.0,
            min_seam_score=0.0,
            min_motion_score=0.0,
            min_flicker_score=0.0,
            min_frame_quality=0.0,
        )
    )
    frames = [_gradient_frame(i * 15) for i in range(8)]
    result = evaluator.evaluate(frames)
    assert result.accepted


def test_evaluator_produces_metrics() -> None:
    evaluator = TemporalQualityEvaluator()
    frames = [_gradient_frame(i * 20) for i in range(6)]
    result = evaluator.evaluate(frames)
    assert "seam_score" in result.metrics
    assert "motion_score" in result.metrics
    assert "flicker_score" in result.metrics
    assert "min_frame_quality" in result.metrics
    assert "luminance_variation" in result.metrics
    assert "color_variation" in result.metrics


def test_evaluator_seam_check_with_virtual_frames() -> None:
    evaluator = TemporalQualityEvaluator(
        TemporalQualityThresholds(min_seam_score=0.99)
    )
    frames = [_gradient_frame(i * 10) for i in range(4)]
    identical_frame = frames[0].copy()
    result = evaluator.evaluate(frames, virtual_phase1=identical_frame)
    assert result.seam_score == pytest.approx(1.0, abs=1e-6)
