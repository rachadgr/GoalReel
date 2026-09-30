"""Adapter détection — Ultralytics YOLO (YOLOv8s COCO par défaut).

Fonction classe-agnostique : les noms d'objets utiles au football sont
mappés (person=0, sports ball=32). Un checkpoint football custom ``best.pt``
peut être branché via ``GOALREEL_DETECTOR_WEIGHTS`` sans changer le code.

Vérité : ce checkpoint (``yolov8s.pt``) est un modèle **COCO générique**, il
n'est **pas** entraîné football. On expose donc explicitement les classes
réelles ; ``referee`` n'existe pas dans COCO et n'est jamais inventé.
"""
from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable

# COCO ids utiles au football
PLAYER_CLASS_IDS = (0,)        # person
BALL_CLASS_IDS = (32, 37)      # sports ball, surfboard (ballon blanc)

# Classes football que le modèle ne fournit PAS (jamais inventées)
FOOTBALL_TARGETS = ("player", "ball", "referee", "goal", "goalkeeper")


def _class_report(names: dict[int, str]) -> dict[str, Any]:
    """Analyse honnête des classes réellement présentes dans le checkpoint."""
    values = {str(v).lower() for v in (names or {}).values()}
    fname = [str(names[i]).lower() for i in (names or {})]
    report: dict[str, Any] = {
        "num_classes": len(names or {}),
        "has_person": "person" in values,
        "has_sports_ball": "sports ball" in values,
        "has_referee": any("referee" in v for v in fname),
        "football_trained": any(v in values for v in FOOTBALL_TARGETS),
    }
    report["missing_football_classes"] = [
        t for t in FOOTBALL_TARGETS
        if not any(t in v for v in fname)
    ]
    return report


def load_detector(spec: ModelSpec, manager: Any):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - dépendance optionnelle
        raise ModelUnavailable(f"ultralytics unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc

    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("YOLO checkpoint missing",
                               truth="CHECKPOINT_MISSING")

    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise ModelUnavailable(f"YOLO checkpoint unreadable: {exc}",
                               truth="INCOMPATIBLE") from exc

    device = manager.state(spec.name).device
    try:
        model.to(device if device != "auto" else "cpu")
    except Exception:
        pass

    names = getattr(model, "names", {}) or {}
    class_report = _class_report(names)
    # On n'annonce jamais un modèle football alors qu'il ne l'est pas.
    manager.state(spec.name).meta.update({
        "checkpoint": str(path),
        "num_classes": class_report["num_classes"],
        "football_trained": class_report["football_trained"],
        "class_report": class_report,
    })

    class _Detector:
        def __init__(self):
            self.model = model
            self.names = names
            self.conf = spec.confidence or 0.25
            self.imgsz = spec.input_size or 640
            self.class_report = class_report

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

    tag = "COCO/generic" if not class_report["football_trained"] else "football"
    return _Detector(), f"YOLO loaded ({class_report['num_classes']} classes, {tag})"
