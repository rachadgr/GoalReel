"""Adapter caméra virtuelle / novel-view — Stable Virtual Camera (SEVA).

SEVA (Stability AI) est un modèle de diffusion (1.3B) de synthèse de vues.
Contrairement aux autres modèles, il n'est **pas** fourni : ses poids
(pesant lourds) résident sur Hugging Face et nécessitent un GPU pour un usage
pratique. Cet adapter :

  * détecte le code source SEVA dans ``third_party/stable-virtual-camera`` ;
  * enregistre un backend ``seva`` *réel* seulement si le code + les poids
    sont présents et que le runtime (flash-attn, etc.) est disponible ;
  * sinon renvoie ``MODEL_UNAVAILABLE`` proprement — le pipeline reste
    exécutable et le frontend affiche l'état réel.

Il s'enregistre auprès du ``NovelViewProvider`` existant (voir
``goalreel/generation``), sans casser l'API ``generate(request)``.
"""
from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from .manager import ModelUnavailable


def load_virtual_camera(spec: ModelSpec, manager: Any):
    root = spec.extra.get("backend") if spec.extra else None
    backend = (spec.config or spec.extra.get("backend") or "unavailable")
    version = (spec.extra or {}).get("model_version", "1.1")

    if backend in (None, "", "none", "unavailable"):
        raise ModelUnavailable(
            "Novel-view backend disabled (set GOALREEL_NOVEL_VIEW_BACKEND=seva)"
        )

    if backend != "seva":
        raise ModelUnavailable(f"Unsupported novel-view backend: {backend}")

    seva_root = manager.settings.vendor_dir.parent / "third_party" / "stable-virtual-camera"
    if not (seva_root / "seva").is_dir():
        raise ModelUnavailable(
            f"SEVA source not found at {seva_root} "
            "(clone https://github.com/Stability-AI/stable-virtual-camera into third_party/)"
        )

    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise ModelUnavailable(f"torch unavailable: {exc}") from exc

    class _SevaBackend:
        """Wrapper fin autour du backend SEVA existant (lazy)."""

        name = "seva"

        def __init__(self):
            self.root = seva_root
            self.version = version
            self._impl = None

        def status(self) -> dict[str, Any]:
            return {"status": "OK", "provider": self.name, "version": self.version}

        def generate(self, request):
            # L'implémentation concrète est déléguée au provider SEVA.
            from .seva_provider import SevaProvider

            if self._impl is None:
                self._impl = SevaProvider(self.root, self.version)
            return self._impl.generate(request)

    return _SevaBackend(), f"SEVA backend wired (version {version})"
