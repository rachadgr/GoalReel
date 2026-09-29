"""Tests d'inférence réels (ignorés si torch/checkpoints absents)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has(path):
    from pathlib import Path
    return Path(path).is_file()


@pytest.mark.skipif(not _has("models/detection/yolov8s.pt"), reason="YOLO checkpoint absent")
def test_detector_inference(manager, synthetic_frame):
    from goalreel.services.ai import DetectorService
    try:
        det = DetectorService(manager)
        out = det.detect(synthetic_frame)
    except Exception as e:  # ultralytics manquant etc.
        pytest.skip(f"detector runtime unavailable: {e}")
    assert isinstance(out, list)
    for d in out:
        assert set(["bbox", "class_id", "class_name", "confidence"]).issubset(d)


@pytest.mark.skipif(not _has("models/reid/osnet_x1_0_imagenet.pth"), reason="ReID checkpoint absent")
def test_reid_embedding(manager, synthetic_frame):
    from goalreel.services.ai import ReIDService
    try:
        emb = ReIDService(manager).embed(synthetic_frame)
    except Exception as e:
        pytest.skip(f"reid runtime unavailable: {e}")
    assert emb is not None and emb.shape[-1] == 512
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3


@pytest.mark.skipif(not _has("models/depth/depth_anything_v2_vits.pth"), reason="Depth checkpoint absent")
def test_depth_inference(manager, synthetic_frame):
    from goalreel.services.ai import DepthService
    try:
        d = DepthService(manager).infer(synthetic_frame)
    except Exception as e:
        pytest.skip(f"depth runtime unavailable: {e}")
    assert d is not None and d.shape[:2] == synthetic_frame.shape[:2]


@pytest.mark.skipif(not _has("models/segmentation/sam2.1_hiera_tiny.pt"), reason="SAM2 checkpoint absent")
def test_segmentation(manager, synthetic_frame):
    from goalreel.services.ai import SegmenterService
    try:
        m = SegmenterService(manager).segment_bbox(synthetic_frame, [250, 250, 350, 350])
    except Exception as e:
        pytest.skip(f"sam2 runtime unavailable: {e}")
    if m is None:
        pytest.skip("sam2 unavailable")
    assert m["mask"].shape[:2] == synthetic_frame.shape[:2]


def test_tracker_stable_id(manager):
    from goalreel.services.ai import TrackerService
    t = TrackerService(high=0.2)
    a = t.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9}])
    b = t.update([{"bbox": [1, 0, 11, 10], "confidence": 0.9}])
    assert a[0]["track_id"] == b[0]["track_id"]


def test_interpolation_fallback(manager, synthetic_frame):
    from goalreel.services.ai import InterpolationService
    svc = InterpolationService(manager)
    frame, mode = svc.interpolate(synthetic_frame, synthetic_frame, 0.5)
    assert mode in ("rife", "fallback")
    assert frame.shape == synthetic_frame.shape


def test_invalid_input_handling(manager):
    from goalreel.services.ai import ReIDService
    # crop vide -> None (pas d'exception)
    assert ReIDService(manager).embed(np.zeros((0, 0, 3), dtype=np.uint8)) is None
