"""Adapter détection — Ultralytics YOLO (YOLOv8s COCO par défaut).

Fonction classe-agnostique : les noms d'objets utiles au football sont
mappés (person=0, sports ball=32). Un checkpoint football custom ``best.pt``
peut être branché via ``GOALREEL_DETECTOR_WEIGHTS`` sans changer le code.
"""
from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable

# COCO ids utiles au football
PLAYER_CLASS_IDS = (0,)        # person
BALL_CLASS_IDS = (32, 37)      # sports ball, surfboard (ballon blanc)


def load_detector(spec: ModelSpec, manager: Any):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - dépendance optionnelle
        raise ModelUnavailable(f"ultralytics unavailable: {exc}") from exc

    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("YOLO checkpoint missing")

    model = YOLO(str(path))
    device = manager.state(spec.name).device
    try:
        model.to(device if device != "auto" else "cpu")
    except Exception:
        pass

    class _Detector:
        def __init__(self):
            self.model = model
            self.names = getattr(model, "names", {}) or {}
            self.conf = spec.confidence or 0.25
            self.imgsz = spec.input_size or 640

        def detect(self, frame, conf: float | None = None) -> list[dict[str, Any]]:
            r = self.model.predict(frame, conf=conf or self.conf,
                                   imgsz=self.imgsz, verbose=False)[0]
            out = []
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.cls.cpu().numpy(),
                               r.boxes.conf.cpu().numpy()):
                cid = int(c)
                out.append({
                    "bbox": [float(x) for x in b],
                    "class_id": cid,
                    "class_name": self.names.get(cid, str(cid)),
                    "confidence": float(s),
                })
            return out

        def players(self, frame, conf: float | None = None):
            return [d for d in self.detect(frame, conf) if d["class_id"] in PLAYER_CLASS_IDS]

        def ball(self, frame, conf: float | None = None):
            return [d for d in self.detect(frame, conf) if d["class_id"] in BALL_CLASS_IDS]

    return _Detector(), f"YOLO loaded ({len(getattr(model, 'names', {}))} classes)"
