"""Adapter Re-ID — OSNet x1.0 (backbone vendu depuis torchreid).

Le checkpoint ``osnet_x1_0_imagenet.pth`` est un OSNet x1.0 entraîné sur
ImageNet (backbones OSNet + ``fc`` + ``classifier``). On construit
l'architecture OSNet x1.0 complète puis on charge les poids de façon
tolérante : les couches backbone sont partagées avec la version Market-1501,
ce qui donne un extracteur d'embeddings 512-d exploitable pour la
ré-identification (similarité cosinus).

Optionnel : un vrai checkpoint Market-1501 peut être fourni (mêmes clés).
"""
from __future__ import annotations

import sys
from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable


def _load_osnet(manager: Any):
    vendor = manager.settings.vendor_dir / "osnet"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        import importlib

        # le fichier vendorisé s'appelle osnet.py
        m = importlib.import_module("osnet")
        return m
    except Exception as exc:
        raise ModelUnavailable(f"OSNet architecture unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc


def load_reid(spec: ModelSpec, manager: Any):
    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("Re-ID checkpoint missing",
                               truth="CHECKPOINT_MISSING")
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T
    except Exception as exc:
        raise ModelUnavailable(f"torch/torchvision unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc

    osnet_mod = _load_osnet(manager)
    device = manager.state(spec.name).device

    # num_classes arbitraire : la tête fc est chargée de façon tolérante
    try:
        model = osnet_mod.osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
    except Exception as exc:
        raise ModelUnavailable(f"OSNet build failed: {exc}",
                               truth="DEPENDENCY_MISSING") from exc
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelUnavailable(f"Re-ID checkpoint unreadable: {exc}",
                               truth="INCOMPATIBLE") from exc
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as exc:
        raise ModelUnavailable(f"Re-ID architecture mismatch: {exc}",
                               truth="INCOMPATIBLE") from exc
    model = model.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((spec.input_size or 256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _ReID:
        def __init__(self):
            self.model = model
            self.device = device

        @torch.no_grad()
        def embed(self, crop_bgr):
            """Extrait un embedding L2-normalisé (512-d) d'un crop BGR."""
            import numpy as np

            if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
                return None
            tensor = transform(crop_bgr).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = F.normalize(feat.flatten(1), p=2, dim=1)
            return feat[0].cpu().numpy().astype("float32")

        @staticmethod
        def cosine(a, b) -> float:
            import numpy as np

            a = np.asarray(a); b = np.asarray(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
            return float(np.dot(a, b) / denom)

        def _raw_model(self):
            return self.model

    total = len(state) or 1
    coverage = (total - len(missing)) / total
    manager.state(spec.name).meta.update({
        "checkpoint": str(path),
        "backbone": "osnet_x1_0",
        "pretraining": "imagenet (NOT Market-1501 / NOT football-specific)",
        "embedding_dim": 512,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "coverage": round(coverage, 4),
    })
    if coverage < 0.9:
        raise ModelUnavailable(
            f"Re-ID checkpoint incompatible with OSNet x1.0 "
            f"(coverage={coverage:.2f})",
            truth="INCOMPATIBLE",
        )
    msg = f"OSNet x1.0 loaded; missing={len(missing)} unexpected={len(unexpected)}"
    return _ReID(), msg
