"""Adapter profondeur — Depth Anything V2 (ViT-S, vits).

Architecture : code source *vendorisé* (``vendor/depth_anything_v2``) afin de
ne dépendre d'aucun package externe. Les poids sont le state_dict natif
(``pretrained.*`` DINOv2 + ``depth_head.*``). Fallback vers ``timm`` si
nécessaire.
"""
from __future__ import annotations

import sys
from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable

# Encoder vits = ViT-S/14
_VITS_KWARGS = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])


def _load_arch(manager: Any):
    """Importe DepthAnythingV2 depuis le vendor, sinon via timm."""
    vendor = manager.settings.vendor_dir
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        return DepthAnythingV2, "vendored"
    except Exception:
        pass
    try:
        import importlib

        m = importlib.import_module("depth_anything_v2.dpt")
        return m.DepthAnythingV2, "installed"
    except Exception as exc:
        raise ModelUnavailable(f"DepthAnythingV2 code unavailable: {exc}") from exc


def load_depth(spec: ModelSpec, manager: Any):
    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("Depth Anywhere V2 checkpoint missing")
    try:
        import torch
    except Exception as exc:
        raise ModelUnavailable(f"torch unavailable: {exc}") from exc

    DepthAnythingV2, src = _load_arch(manager)
    device = manager.state(spec.name).device

    model = DepthAnythingV2(**_VITS_KWARGS)
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("pretrained.") for k in state
    ):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    class _Depth:
        def __init__(self):
            self.model = model
            self.device = device
            self.input_size = spec.input_size or 518

        @torch.no_grad()
        def infer(self, image_bgr):
            """Renvoie une carte de profondeur float32 (HxW), valeurs relatives."""
            depth = self.model.infer_image(image_bgr, self.input_size)
            return depth

        def _raw_model(self):
            return self.model

    msg = f"Depth Anything V2 loaded ({src}); missing={len(missing)} unexpected={len(unexpected)}"
    return _Depth(), msg
