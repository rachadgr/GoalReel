"""Adapter pose — estimation de posture.

Aucun checkpoint de pose n'est fourni. Le module reste donc
``MODEL_UNAVAILABLE`` par défaut, mais l'interface est prête : fournir un
checkpoint (ex. YOLOv8-pose via ``GOALREEL_POSE_WEIGHTS``) l'active sans
modifier le pipeline.
"""
from __future__ import annotations

from ..config import ModelSpec
from .manager import ModelUnavailable


def load_pose(spec: ModelSpec, manager):
    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("No pose checkpoint configured")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ModelUnavailable(f"ultralytics unavailable: {exc}") from exc

    model = YOLO(str(path))

    class _Pose:
        def __init__(self):
            self.model = model
            self.kpt_names = getattr(model, "kpt_shape", None)

        def estimate(self, frame):
            r = self.model.predict(frame, verbose=False)[0]
            out = []
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()
                conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                for i, k in enumerate(kpts):
                    out.append({
                        "keypoints": k.tolist(),
                        "confidence": conf[i].tolist() if conf is not None else None,
                    })
            return out

    return _Pose(), "Pose model loaded"
