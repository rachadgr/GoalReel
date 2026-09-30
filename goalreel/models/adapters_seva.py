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
    backend = (spec.config or spec.extra.get("backend") or "unavailable")
    version = (spec.extra or {}).get("model_version", "1.1")

    if backend in (None, "", "none", "unavailable"):
        raise ModelUnavailable(
            "Novel-view backend disabled (set GOALREEL_NOVEL_VIEW_BACKEND=seva)",
            truth="DISABLED",
        )

    if backend != "seva":
        raise ModelUnavailable(f"Unsupported novel-view backend: {backend}",
                               truth="INCOMPATIBLE")

    seva_root = manager.settings.vendor_dir.parent / "third_party" / "stable-virtual-camera"
    if not (seva_root / "seva").is_dir():
        raise ModelUnavailable(
            f"SEVA source not found at {seva_root} "
            "(clone https://github.com/Stability-AI/stable-virtual-camera into third_party/)",
            truth="CHECKPOINT_MISSING",
        )

    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise ModelUnavailable(f"torch unavailable: {exc}",
                               truth="DEPENDENCY_MISSING") from exc

    # SEVA est un modèle de diffusion lourd (1.3B). En CPU, il n'est pas
    # exploitable en pratique : on rapporte honnêtement GPU_REQUIRED plutôt
    # que de laisser croire à une disponibilité de fait.
    device = manager.state(spec.name).device
    gpu_ok = device.startswith("cuda") and torch.cuda.is_available()

    # Les poids HF (stabilityai/stable-virtual-camera) ne sont pas fournis et
    # ne sont pas téléchargés automatiquement (volumineux). Un chemin explicite
    # peut être fourni via GOALREEL_SEVA_WEIGHTS pour un runtime réel.
    weights = spec.resolved_path()
    weights_present = bool(weights and weights.is_file())

    manager.state(spec.name).meta.update({
        "code_present": True,
        "weights_provided": weights_present,
        "weights_path": str(weights) if weights else None,
        "device": device,
        "gpu_available": gpu_ok,
        "stage": "CODE_INTEGRATED",
    })

    # États honnêtes : jamais READY sans poids + GPU. Le pipeline continue avec
    # le backend « unavailable » et le reste du pipeline reste exécutable.
    if not weights_present:
        raise ModelUnavailable(
            "SEVA weights not provided "
            "(stable-virtual-camera HF checkpoint required; set GOALREEL_SEVA_WEIGHTS)",
            truth="CHECKPOINT_MISSING",
        )
    if not gpu_ok:
        raise ModelUnavailable(
            "SEVA weights present but no CUDA GPU available (GPU_REQUIRED)",
            truth="GPU_REQUIRED",
        )

    class _SevaBackend:
        """Wrapper fin autour du backend SEVA existant (lazy)."""

        name = "seva"

        def __init__(self):
            self.root = seva_root
            self.version = version
            self._impl = None

        def status(self) -> dict[str, Any]:
            return {"status": "OK", "provider": self.name, "version": self.version,
                    "device": device, "gpu": gpu_ok}

        def generate(self, request):
            # L'implémentation concrète est déléguée au provider SEVA.
            from .seva_provider import SevaProvider

            if self._impl is None:
                self._impl = SevaProvider(self.root, self.version)
            return self._impl.generate(request)

    return _SevaBackend(), f"SEVA backend wired (version {version}, device={device})"
