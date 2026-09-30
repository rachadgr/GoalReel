"""Adapter segmentation — SAM 2.1 (Hiera).

Le checkpoint ``sam2.1_hiera_tiny.pt`` provient de Meta. Le chargement réel
passe par le paquet ``sam2`` (build_sam2 + _load_checkpoint) puis
``SAM2ImagePredictor`` pour l'inférence promptée (point/box -> masque).
"""
from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable

# configs disponibles avec le paquet sam2
_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _resolve_config(spec: ModelSpec) -> str:
    cfg = spec.config or "sam2.1_hiera_t"
    if cfg.endswith(".yaml"):
        return cfg
    return f"configs/sam2.1/{cfg}.yaml"


def load_segmenter(spec: ModelSpec, manager: Any):
    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("SAM2 checkpoint missing",
                               truth="CHECKPOINT_MISSING")

    # API récente (SAM 2.1) : sam2.sam2_image_predictor
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
    except Exception as exc:
        raise ModelUnavailable(f"sam2 runtime unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc

    device = manager.state(spec.name).device
    if device.startswith("cuda"):
        device = "cuda" if device == "cuda" else device

    cfg = _resolve_config(spec)
    try:
        model = build_sam2(cfg, str(path), device=device, apply_postprocessing=False)
        predictor = SAM2ImagePredictor(model)
    except Exception as exc:
        # Le checkpoint est présent mais l'architecture/config ne s'apparie pas
        # (ou une dépendance interne manque) : state honnête, pas de faux OK.
        raise ModelUnavailable(f"sam2 init failed: {exc}",
                               truth="INCOMPATIBLE") from exc

    manager.state(spec.name).meta.update({
        "checkpoint": str(path),
        "config": cfg,
        "runtime": "sam2",
    })

    class _Segmenter:
        def __init__(self):
            self.model = model
            self.predictor = predictor
            self.device = device

        def set_image(self, image_bgr):
            import cv2

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb)

        def segment_from_box(self, box_xyxy) -> Any:
            import numpy as np

            masks, scores, _ = self.predictor.predict(
                box=np.asarray(box_xyxy, dtype=np.float32)[None, :],
                multimask_output=False,
            )
            return {"mask": masks[0], "score": float(scores[0])}

        def segment_from_points(self, points, labels) -> Any:
            import numpy as np

            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=False,
            )
            return {"mask": masks[0], "score": float(scores[0])}

    return _Segmenter(), f"SAM2 predictor loaded (cfg={cfg})"
