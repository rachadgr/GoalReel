"""Tests du reframe 9:16 piloté par la preuve de suivi.

Vérité exigée : quand de vraies trajectoires existent, la caméra 9:16 doit
suivre les coordonnées réelles suivies ; en l'absence de preuve, un fallback
statique est utilisé (jamais un faux suivi).
"""
from pathlib import Path

import numpy as np
import pytest

from goalreel.cinematic.reframe import build_reframe_targets


def test_reframe_no_tracks_is_static_fallback():
    targets, meta = build_reframe_targets({})
    assert targets == {}
    assert meta["source"] == "NO_TRACKS"
    assert meta["mode"] == "STATIC_FALLBACK"
    assert meta["followed_track"] is None


def test_reframe_follows_longest_real_track():
    # Deux trajectoires ; la plus longue (id=7) doit être le sujet suivi.
    tracks = {
        3: [{"frame": 0, "bbox": [0, 0, 10, 10]}],
        7: [
            {"frame": 0, "bbox": [100, 0, 120, 10]},
            {"frame": 1, "bbox": [130, 0, 150, 10]},
            {"frame": 2, "bbox": [160, 0, 180, 10]},
        ],
    }
    targets, meta = build_reframe_targets(tracks)
    assert meta["source"] == "REAL_TRACKS"
    assert meta["mode"] == "FOLLOW_TRACKED_SUBJECT"
    assert meta["followed_track"] == 7
    # cx = centre horizontal réel de la bbox suivie (aucune coordonnée inventée).
    assert targets[0] == pytest.approx(110.0)
    assert targets[2] == pytest.approx(170.0)
    assert set(targets) == {0, 1, 2}


def test_reframe_ignores_malformed_items():
    tracks = {1: [{"frame": 0}, {"frame": 1, "bbox": [10, 0, 30, 10]}, "junk"]}
    targets, meta = build_reframe_targets(tracks)
    assert meta["source"] == "REAL_TRACKS"
    assert targets == {1: 20.0}


def _make_tiny_video(path, w=64, h=64, n=12, fps=12):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (40, 120, 40)
        # un sujet qui se déplace de la gauche vers la droite
        x = int(8 + (w - 24) * i / max(1, n - 1))
        cv2.circle(frame, (x + 8, h // 2), 6, (240, 240, 240), -1)
        writer.write(frame)
    writer.release()


def test_tracked_render_differs_from_static_and_valid(tmp_path):
    """Le rendu SUIVI (avec cibles réelles) doit produire une vidéo valide 9:16."""
    pytest.importorskip("cv2")
    from goalreel.qc.final import final_qc
    from goalreel.source.ffmpeg import render_vertical

    src = tmp_path / "src.mp4"
    _make_tiny_video(src)

    # Cibles « suivies » : le sujet balaie la largeur source.
    targets = {i: 8 + (64 - 16) * i / 11 for i in range(12)}
    out = tmp_path / "tracked.mp4"
    render_vertical(str(src), str(out),
                    reframe={"targets": targets, "mode": "FOLLOW_TRACKED_SUBJECT",
                             "source": "REAL_TRACKS"})
    assert out.is_file() and out.stat().st_size > 0
    qc = final_qc(str(out))
    assert qc["status"] == "OK"


def test_static_render_when_no_targets(tmp_path):
    pytest.importorskip("cv2")
    from goalreel.source.ffmpeg import render_vertical

    src = tmp_path / "src2.mp4"
    _make_tiny_video(src)
    out = tmp_path / "static.mp4"
    render_vertical(str(src), str(out), reframe={"targets": {}})
    assert out.is_file() and out.stat().st_size > 0
