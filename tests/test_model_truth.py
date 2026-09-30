"""Tests — vérité du runtime réel des modèles (Phase 10) et provisioning.

Ces tests verrouillent les garanties suivantes :

  * le vocabulaire de vérité est strict et jamais simulé ;
  * un modèle présent sur disque est correctement rapporté ``CHECKPOINT_PRESENT``
    et, quand le runtime existe, ``MODEL_LOADS`` + ``INFERENCE_WORKS`` ;
  * l'absence d'un checkpoint est honnêtement ``CHECKPOINT_MISSING`` (jamais OK) ;
  * ``GPU_VALIDATED`` n'est JAMAIS revendiqué sans CUDA ;
  * SAM 2.1 n'est déclaré vérifié que si un masque réel est produit ;
  * le script de provisioning couvre bien tous les checkpoints attendus.

Les tests qui dépendent de poids réellement présents sont *skippés proprement*
quand l'environnement est une copie fraîche sans checkpoints (CI), sans jamais
affaiblir la garantie quand les poids sont là.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "assets" / "SOURCE_MASTER_1000006866.mp4"

DETECTOR = ROOT / "models/detection/yolov8s.pt"
DEPTH = ROOT / "models/depth/depth_anything_v2_vits.pth"
REID = ROOT / "models/reid/osnet_x1_0_imagenet.pth"
SAM2 = ROOT / "models/segmentation/sam2.1_hiera_tiny.pt"


def _frames(n=2):
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    out, idx, kept = [], 0, 0
    while kept < n:
        ok, f = cap.read()
        if not ok:
            break
        if idx % 30 == 0:
            out.append((idx, f))
            kept += 1
        idx += 1
    cap.release()
    return out


# --------------------------------------------------------------------------- #
# Vocabulaire de vérité (statique, jamais simulé)
# --------------------------------------------------------------------------- #
def test_truth_vocabulary_is_exposed():
    from goalreel.models import verify

    for token in ("CODE_INTEGRATED", "CHECKPOINT_PRESENT", "MODEL_LOADS",
                  "INFERENCE_WORKS", "GPU_VALIDATED", "END_TO_END_VALIDATED"):
        assert getattr(verify, token) == token


def test_no_gpu_validated_without_cuda(manager):
    """GPU_VALIDATED ne doit jamais apparaître sans CUDA."""
    from goalreel.config import detect_device

    has_cuda = detect_device("auto") == "cuda"
    if has_cuda:
        pytest.skip("CUDA présent : la garantie négative ne s'applique pas")
    # Construit des labels pour un modèle chargé : aucun GPU_VALIDATED.
    from goalreel.models.verify import _labels

    labels = _labels(manager, "detector", loads=True, inference=True, e2e=True)
    assert "GPU_VALIDATED" not in labels


# --------------------------------------------------------------------------- #
# YOLO : détection réelle (personne) + honnêteté COCO
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not DETECTOR.is_file(), reason="YOLO checkpoint absent")
def test_detector_real_inference_verified(manager):
    if not VIDEO.is_file():
        pytest.skip("source video absent")
    from goalreel.models.verify import verify_detector

    rec = verify_detector(manager, _frames())
    assert rec["verified"] is True
    assert "INFERENCE_WORKS" in rec["labels"]
    assert rec["evidence"]["objects"] > 0
    # COCO générique : jamais football-trained, jamais referee/goal inventés.
    report = rec["evidence"]["class_report"]
    assert report["football_trained"] is False
    assert report["has_referee"] is False


def test_detector_missing_checkpoint_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("GOALREEL_DETECTOR_WEIGHTS", str(tmp_path / "nope.pt"))
    from goalreel.config import load_settings
    from goalreel.models.manager import ModelManager
    from goalreel.models.verify import verify_detector

    mgr = ModelManager(settings=load_settings())
    rec = verify_detector(mgr, [])
    assert rec["verified"] is False
    assert "CHECKPOINT_PRESENT" not in rec["labels"]


# --------------------------------------------------------------------------- #
# OSNet : embedding 512-d réel, L2-normalisé
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REID.is_file(), reason="ReID checkpoint absent")
def test_reid_real_embedding_verified(manager):
    if not VIDEO.is_file():
        pytest.skip("source video absent")
    from goalreel.models.verify import verify_reid

    rec = verify_reid(manager, _frames())
    assert rec["verified"] is True
    assert rec["evidence"]["embedding_dim"] == 512
    assert rec["evidence"]["l2_normalized"] is True
    assert rec["evidence"]["l2_norm"] > 0.0


# --------------------------------------------------------------------------- #
# Depth Anything V2 : carte réelle HxW numérique
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not DEPTH.is_file(), reason="Depth checkpoint absent")
def test_depth_real_inference_verified(manager):
    if not VIDEO.is_file():
        pytest.skip("source video absent")
    from goalreel.models.verify import verify_depth

    rec = verify_depth(manager, _frames())
    assert rec["verified"] is True
    h, w = rec["evidence"]["shape"]
    assert h > 0 and w > 0
    assert rec["evidence"]["max"] > rec["evidence"]["min"]


# --------------------------------------------------------------------------- #
# SAM 2.1 : vérifié UNIQUEMENT si un masque réel est produit
# --------------------------------------------------------------------------- #
def test_sam2_status_is_truthful(manager):
    if not SAM2.is_file():
        pytest.skip("SAM2 checkpoint absent")
    from goalreel.models.verify import verify_sam2

    rec = verify_sam2(manager, _frames())
    if rec["verified"]:
        assert "INFERENCE_WORKS" in rec["labels"]
        assert rec["evidence"]["mask_pixels"] > 0
    else:
        # Sans runtime, l'état doit être honnête (jamais MODEL_LOADS).
        assert "MODEL_LOADS" not in rec["labels"]


# --------------------------------------------------------------------------- #
# Provisioning : le script couvre tous les checkpoints attendus
# --------------------------------------------------------------------------- #
def test_download_script_covers_expected_checkpoints():
    script = (ROOT / "scripts/download_models.sh").read_text(encoding="utf-8")
    for fname in ("yolov8s.pt", "sam2.1_hiera_tiny.pt",
                  "depth_anything_v2_vits.pth", "osnet_x1_0_imagenet.pth"):
        assert fname in script, f"{fname} absent du provisioning"
    # OSNet est désormais provisionnable automatiquement (gdown documenté).
    assert "gdown" in script
    assert "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY" in script


# --------------------------------------------------------------------------- #
# Agrégat : verify_all renvoie un rapport cohérent et sérialisable
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (DETECTOR.is_file() and VIDEO.is_file()),
                    reason="modèles/source absents")
def test_verify_all_report_is_json_serializable(manager):
    from goalreel.models.verify import verify_all

    report = verify_all(str(VIDEO), manager=manager, every=60, samples=1)
    assert report["schema"] == "goalreel.model_truth.v1"
    # Sérialisable en JSON strict (aucun type numpy ne doit fuir).
    json.dumps(report)
    assert "detector" in report["summary"]
    # Sans CUDA, gpu_validated est False.
    if not report["cuda"].get("available"):
        assert report["gpu_validated"] is False


def test_verify_all_never_claims_fake_readiness(tmp_path, monkeypatch):
    """Sans aucun checkpoint, aucun modèle ne doit être 'VERIFIED'."""
    for env, fname in (("GOALREEL_DETECTOR_WEIGHTS", "d.pt"),
                       ("GOALREEL_DEPTH_CHECKPOINT", "d.pth"),
                       ("GOALREEL_REID_WEIGHTS", "r.pth"),
                       ("GOALREEL_SAM2_CHECKPOINT", "s.pt")):
        monkeypatch.setenv(env, str(tmp_path / fname))
    from goalreel.config import load_settings
    from goalreel.models.manager import ModelManager
    from goalreel.models.verify import verify_all

    mgr = ModelManager(settings=load_settings())
    report = verify_all(str(VIDEO), manager=mgr, every=120, samples=1)
    assert all(not m["verified"] for m in report["models"].values())
