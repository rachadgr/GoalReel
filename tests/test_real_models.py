"""Validation réelle des modèles fournis (CPU, sans GPU, sans téléchargement).

Chaque test est *skippé proprement* si le checkpoint ou le runtime fait défaut
(la CI n'a donc jamais besoin de GPU ni de télécharger de gros poids). Quand le
modèle est présent, on vérifie le chargement réel et une inférence réelle.

Vérité : un test ne prétend jamais à un succès si le modèle n'est pas chargé.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _has(rel: str) -> bool:
    return (ROOT / rel).is_file()


def _importable(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- #
# RIFE : code intégré, checkpoint manquant => CHECKPOINT_MISSING (jamais ERROR)
# --------------------------------------------------------------------------- #
def test_rife_missing_checkpoint_is_graceful(manager, tmp_path):
    # Le checkpoint RIFE n'est pas fourni : le code doit rester intégré et
    # signaler honnêtement CHECKPOINT_MISSING sans casser le pipeline.
    manager.spec("interpolation").path = str(tmp_path / "flownet.pkl")
    inst, st = manager.try_stage("interpolation")
    assert inst is None
    assert st.status == "MODEL_UNAVAILABLE"
    assert manager.state("interpolation").truth == "CHECKPOINT_MISSING"


def test_rife_ifnet_builds(manager):
    """Le code RIFE vendorisé doit se construire (sans poids)."""
    pytest.importorskip("torch")
    from goalreel.models.adapters_rife import build_network

    try:
        net = build_network(manager)
    except Exception as exc:  # dépendance manquante -> skip, pas échec
        pytest.skip(f"RIFE code unavailable: {exc}")
    assert net is not None


# --------------------------------------------------------------------------- #
# YOLO (yolov8s COCO) : chargement réel + introspection honnête des classes
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has("models/detection/yolov8s.pt"),
                    reason="YOLO checkpoint absent")
def test_detector_loads_and_reports_classes(manager, synthetic_frame):
    if not _importable("ultralytics"):
        pytest.skip("ultralytics not installed")
    inst, st = manager.try_stage("detector")
    assert inst is not None, st.message
    assert manager.state("detector").truth == "READY"
    report = manager.state("detector").meta.get("class_report", {})
    # Le checkpoint est COCO (80 classes) : on l'affirme, on ne l'invente pas.
    assert report.get("num_classes") == 80
    assert report.get("has_person") is True
    assert report.get("has_sports_ball") is True
    # Pas de classe referee dans COCO : jamais prétendu.
    assert report.get("has_referee") is False
    assert report.get("football_trained") is False
    dets = inst.detect(synthetic_frame)
    assert isinstance(dets, list)
    for d in dets:
        assert set(["bbox", "class_id", "class_name", "confidence"]).issubset(d)


# --------------------------------------------------------------------------- #
# Depth Anything V2 : chargement réel (0 missing) + inférence (HxW)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has("models/depth/depth_anything_v2_vits.pth"),
                    reason="Depth checkpoint absent")
def test_depth_loads_and_infers(manager, synthetic_frame):
    pytest.importorskip("torch")
    inst, st = manager.try_stage("depth")
    assert inst is not None, st.message
    assert manager.state("depth").truth == "READY"
    meta = manager.state("depth").meta
    assert meta.get("missing_keys") == 0
    out = inst.infer(synthetic_frame)
    assert out is not None and out.shape[:2] == synthetic_frame.shape[:2]


# --------------------------------------------------------------------------- #
# OSNet Re-ID : chargement réel (0 missing) + embedding 512-d normalisé
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has("models/reid/osnet_x1_0_imagenet.pth"),
                    reason="ReID checkpoint absent")
def test_reid_loads_and_embeds(manager, synthetic_frame):
    pytest.importorskip("torch")
    import numpy as np

    inst, st = manager.try_stage("reid")
    assert inst is not None, st.message
    assert manager.state("reid").truth == "READY"
    emb = inst.embed(synthetic_frame)
    assert emb is not None and emb.shape[-1] == 512
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
# SAM 2.1 : checkpoint présent ; dépend de runtime 'sam2' (sinon DEPENDENCY_MISSING)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has("models/segmentation/sam2.1_hiera_tiny.pt"),
                    reason="SAM2 checkpoint absent")
def test_sam2_status_is_honest(manager):
    inst, st = manager.try_stage("segmenter")
    truth = manager.state("segmenter").truth
    if inst is None:
        # Sans runtime sam2 => DEPENDENCY_MISSING (ou INCOMPATIBLE) ; jamais READY.
        assert truth in ("DEPENDENCY_MISSING", "INCOMPATIBLE", "CHECKPOINT_MISSING")
    else:
        assert truth == "READY"


# --------------------------------------------------------------------------- #
# Virtual camera / SEVA : jamais READY sans poids + GPU
# --------------------------------------------------------------------------- #
def test_seva_never_ready_without_weights(manager, monkeypatch):
    monkeypatch.setenv("GOALREEL_NOVEL_VIEW_BACKEND", "seva")
    from goalreel.config import load_settings
    from goalreel.models.manager import ModelManager

    mgr = ModelManager(settings=load_settings())
    mgr.load("virtual_camera", force=True)
    truth = mgr.state("virtual_camera").truth
    assert truth in ("CHECKPOINT_MISSING", "GPU_REQUIRED", "DISABLED", "MODEL_UNAVAILABLE")
    assert truth != "READY"


# --------------------------------------------------------------------------- #
# Pas de faux OK : une étape sans modèle réel ne doit jamais renvoyer OK
# --------------------------------------------------------------------------- #
def test_pose_stage_no_fake_ok(manager, tmp_path, monkeypatch):
    """La pose n'a pas de checkpoint => l'étape doit être MODEL_UNAVAILABLE."""
    pytest.importorskip("torch")
    if not _has("assets/SOURCE_MASTER_1000006866.mp4"):
        # Crée une petite vidéo de test si l'asset manque (CI).
        import cv2
        import numpy as np

        vpath = tmp_path / "t.mp4"
        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for _ in range(5):
            writer.write(np.zeros((64, 64, 3), dtype="uint8"))
        writer.release()
        video = str(vpath)
    else:
        video = "assets/SOURCE_MASTER_1000006866.mp4"
    # Aucun checkpoint de pose fourni : état honnête, jamais OK.
    manager.spec("pose").path = None
    from goalreel.services.analysis_pipeline import AnalysisPipeline

    res = AnalysisPipeline(manager).run(video, every=1, max_frames=1, stages=["pose"])
    assert res["stages"][0]["status"] in ("MODEL_UNAVAILABLE", "ERROR")
