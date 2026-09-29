"""Tests du pipeline cinématique (interpolation / super-résolution)."""
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_slowdown_fallback(manager, synthetic_frame):
    from goalreel.services.ai import InterpolationService

    svc = InterpolationService(manager)
    frames = [synthetic_frame, synthetic_frame, synthetic_frame]
    out = svc.slowdown_frames(frames, factor=2)
    # 3 frames -> 2*(2-1) + 1 = ... dépend du facteur : au moins > in
    assert len(out) >= len(frames)


def test_cinematic_interpolate_writes_clip(manager, tmp_path):
    from goalreel.services.cinematic_pipeline import CinematicPipeline

    frames = [np.full((64, 96, 3), i * 20, dtype=np.uint8) for i in range(5)]
    out = tmp_path / "clip.mp4"
    res = CinematicPipeline(manager).interpolate_clip(frames, str(out), factor=2, fps=15)
    assert res["status"] == "OK"
    assert out.is_file() and out.stat().st_size > 0


def test_cinematic_upscale(manager, tmp_path):
    from goalreel.services.cinematic_pipeline import CinematicPipeline

    frames = [np.full((32, 48, 3), 128, dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "up.mp4"
    res = CinematicPipeline(manager).upscale_clip(frames, str(out), fps=15)
    assert res["status"] == "OK" and res["scale"] == 2
    assert out.is_file()


def test_super_resolution_fallback(manager, synthetic_frame):
    from goalreel.services.ai import SuperResolutionService

    up, mode = SuperResolutionService(manager, scale=2).upscale(synthetic_frame)
    assert mode == "fallback"
    h, w = synthetic_frame.shape[:2]
    assert up.shape[:2] == (h * 2, w * 2)
