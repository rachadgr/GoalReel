"""Test de bout en bout (assets stubbés) — pipeline complet sans GPU."""
from pathlib import Path

import cv2
import numpy as np
import pytest

VIDEO = Path("assets/SOURCE_MASTER_1000006866.mp4")


@pytest.mark.skipif(not VIDEO.is_file(), reason="sample video absent")
def test_full_run_with_stubbed_detector(manager, tmp_path, monkeypatch):
    """E2E : pipeline complet avec un détecteur factice (mock) pour rester rapide."""
    from goalreel.services.ai.detector import DetectorService

    def fake_detect(self, frame, conf=None):
        h, w = frame.shape[:2]
        return [
            {"bbox": [w * 0.4, h * 0.3, w * 0.5, h * 0.7], "class_id": 0,
             "class_name": "person", "confidence": 0.9},
        ]

    monkeypatch.setattr(DetectorService, "detect", fake_detect)
    monkeypatch.setattr(DetectorService, "players", lambda self, f, c=None: self.detect(f))

    import run_goalreel

    report = run_goalreel.run(str(VIDEO), tmp_path / "out", Path("models"),
                              Path("checkpoints"), every=120, manager=manager,
                              analyze=True, max_frames=1, stages=["detection", "tracking"])
    assert "final_reel" in report["outputs"]
    assert (tmp_path / "out" / "final_reel.mp4").is_file()
    assert report["final_qc"]["status"] == "OK"
    assert (tmp_path / "out" / "analysis_report.json").is_file()


def test_run_no_analyze_flag(tmp_path):
    """--no-analyze : ne doit pas échouer même sans modèles."""
    import run_goalreel

    if not VIDEO.is_file():
        pytest.skip("sample video absent")
    report = run_goalreel.run(str(VIDEO), tmp_path / "o2", Path("models"),
                              Path("checkpoints"), every=240, analyze=False)
    assert "backend_summary" in report
