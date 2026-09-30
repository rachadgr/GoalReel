"""Adapter interpolation — RIFE (Real-Time Intermediate Flow Estimation).

RIFE interpole des frames intermédiaires entre deux images (slow-motion fluide
/ montée en fps). Le code du modèle est *vendorisé* (``vendor/rife``) : on
construit ``IFNet`` puis on charge ``flownet.pkl`` si présent.

Un **fallback** de qualité (blend linéaire) est exposé via
``InterpolationService`` lorsque le checkpoint est absent : il ne prétend pas
être du RIFE et l'état du modèle reste ``MODEL_UNAVAILABLE``.
"""
from __future__ import annotations

import sys
from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable

_RIFE_CKPT_NAME = "flownet.pkl"


def _ensure_import_path(manager: Any):
    rife_root = manager.settings.vendor_dir / "rife"
    if str(rife_root) not in sys.path:
        sys.path.insert(0, str(rife_root))


def _load_ifnet_kwargs(spec: ModelSpec) -> dict:
    # IFNet HDv3 standard
    return dict()


def build_network(manager: Any):
    """Construit IFNet (sans poids) - utilisé aussi par le fallback."""
    _ensure_import_path(manager)
    try:
        from model.IFNet import IFNet  # type: ignore

        return IFNet()
    except Exception as exc:
        raise ModelUnavailable(f"RIFE IFNet unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc


def load_interpolation(spec: ModelSpec, manager: Any):
    try:
        import torch
    except Exception as exc:
        raise ModelUnavailable(f"torch unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc

    path = spec.resolved_path()
    device = manager.state(spec.name).device

    net = build_network(manager)
    if path is None or not path.is_file():
        # Le code est intégré (vendor/rife) mais le checkpoint est absent :
        # ce n'est PAS une erreur. On rapporte CHECKPOINT_MISSING et le
        # pipeline continue avec le fallback de blend (jamais présenté comme RIFE).
        raise ModelUnavailable(f"RIFE checkpoint missing ({_RIFE_CKPT_NAME})",
                               truth="CHECKPOINT_MISSING")

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelUnavailable(f"RIFE checkpoint unreadable: {exc}",
                               truth="INCOMPATIBLE") from exc
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # RIFE renomme/strippe 'module.' selon l'entraînement distribué
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # Compatibilité : la majorité des clés du checkpoint doit s'apparier.
    total = len(state) or 1
    coverage = (total - len(missing)) / total
    if coverage < 0.9:
        raise ModelUnavailable(
            f"RIFE checkpoint incompatible (coverage={coverage:.2f}, "
            f"missing={len(missing)}, unexpected={len(unexpected)})",
            truth="INCOMPATIBLE",
        )
    net = net.to(device).eval()
    manager.state(spec.name).meta.update({
        "checkpoint": str(path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "coverage": round(coverage, 4),
    })

    class _Interp:
        def __init__(self):
            self.flownet = net
            self.device = device

        @torch.no_grad()
        def interpolate(self, img0, img1, timestep: float = 0.5):
            """Interpole entre deux frames BGR uint8 -> frame BGR uint8."""
            import numpy as np
            import torch.nn.functional as F

            def to_t(x):
                t = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
                return t[None].to(self.device)

            t0 = to_t(img0)
            t1 = to_t(img1)
            # RIFE travaille sur des dimensions multiples de 32
            h, w = img0.shape[:2]
            ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
            if ph or pw:
                t0 = F.pad(t0, (0, pw, 0, ph))
                t1 = F.pad(t1, (0, pw, 0, ph))
            _, _, merged, _, _, _ = self.flownet(torch.cat((t0, t1), 1), timestep=timestep)
            out = merged[2][:, :, :h, :w].clamp(0, 1)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
            return arr[..., ::-1].copy()

        def _raw_model(self):
            return self.flownet

    msg = f"RIFE IFNet loaded; missing={len(missing)} unexpected={len(unexpected)}"
    return _Interp(), msg
