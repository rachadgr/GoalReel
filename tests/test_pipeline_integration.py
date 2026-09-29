"""Tests d'intégration du pipeline d'analyse et de la robustesse."""
from pathlib import Path

import pytest

VIDEO = Path("assets/SOURCE_MASTER_1000006866.mp4")


def test_pipeline_unknown_stage(manager):
    from goalreel.services.analysis_pipeline import AnalysisPipeline
    res = AnalysisPipeline(manager).run(str(VIDEO), stages=["not_a_stage"])
    assert res["stages"][0]["status"] == "UNKNOWN"


def test_pipeline_handles_missing_models(tmp_path, monkeypatch):
    """Toutes les étapes doivent se terminer (MODEL_UNAVAILABLE compris)."""
    monkeypatch.setenv("GOALREEL_DETECTOR_WEIGHTS", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("GOALREEL_DEPTH_CHECKPOINT", str(tmp_path / "nope.pth"))
    monkeypatch.setenv("GOALREEL_REID_WEIGHTS", str(tmp_path / "nope.pth"))
    monkeypatch.setenv("GOALREEL_SAM2_CHECKPOINT", str(tmp_path / "nope.pt"))
    from goalreel.config import load_settings
    from goalreel.models.manager import ModelManager
    from goalreel.services.analysis_pipeline import AnalysisPipeline

    mgr = ModelManager(settings=load_settings())
    # Vidéo absente : les étapes qui en ont besoin renvoient ERROR proprement.
    res = AnalysisPipeline(mgr).run("does_not_exist.mp4", stages=["detection", "tracking"])
    statuses = {s["name"]: s["status"] for s in res["stages"]}
    assert statuses["detection"] in ("MODEL_UNAVAILABLE", "ERROR")


@pytest.mark.skipif(not VIDEO.is_file(), reason="sample video absent")
def test_detection_stage_real(manager):
    from goalreel.services.analysis_pipeline import AnalysisPipeline
    res = AnalysisPipeline(manager).run(str(VIDEO), every=60, max_frames=1,
                                       stages=["detection", "tracking"])
    by_name = {s["name"]: s for s in res["stages"]}
    # Si YOLO est disponible -> OK, sinon MODEL_UNAVAILABLE (jamais faux OK)
    assert by_name["detection"]["status"] in ("OK", "MODEL_UNAVAILABLE")


@pytest.mark.skipif(not VIDEO.is_file(), reason="sample video absent")
def test_run_analysis_legacy_api(manager):
    import tempfile

    from goalreel.pipeline import run_analysis
    with tempfile.TemporaryDirectory() as d:
        report = run_analysis(str(VIDEO), d, manager=manager, every=60,
                              stages=["tracking"], max_frames=1)
    assert report["schema"] == "goalreel.report.v2"
    assert "model_status" in report
    assert any(s["name"] == "source_analysis" for s in report["stages"])
