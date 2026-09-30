"""ModelManager — couche centralisée d'accès aux modèles GoalReel.

Responsabilités :
  * enregistrement de *loaders* (adapters) par nom de modèle ;
  * chargement paresseux (lazy) et mise en cache (un seul chargement) ;
  * résolution du périphérique (GPU/CPU) et gestion VRAM/RAM best-effort ;
  * déchargement explicite (unload) ;
  * mesure du temps d'inférence ;
  * journalisation structurée ;
  * vérification d'intégrité/compatibilité du checkpoint ;
  * tolérance aux pannes (un modèle indisponible ne casse pas le pipeline).

Le manager ne connaît rien des frameworks : chaque adapter encapsule
ultralytics / torch / sam2 / timm etc. Il expose un état uniforme
(``ModelState``) consommable par l'API et le frontend.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import ModelSpec, Settings, detect_device, load_settings
from ..core.types import StageResult
from .status import (
    READY_TRUTHS,
    UNAVAILABLE_TRUTHS,
    ModelUnavailable,
    pipeline_status,
)

logger = logging.getLogger("goalreel.models")

# État de disponibilité d'un modèle (aligné sur core.types.Status)
UNAVAILABLE = "MODEL_UNAVAILABLE"
OK = "OK"
ERROR = "ERROR"

# États *véridiques* (voir goalreel/models/status.py). Le champ ``ModelState.truth``
# porte le détail honnête ; ``ModelState.status`` reste l'état pipeline historique
# (rétro-compatibilité API/frontend/tests).
CHECKPOINT_MISSING = "CHECKPOINT_MISSING"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
INCOMPATIBLE = "INCOMPATIBLE"
GPU_REQUIRED = "GPU_REQUIRED"
LOAD_ERROR = "LOAD_ERROR"
READY = "READY"
DISABLED = "DISABLED"

__all__ = ["ModelManager", "ModelState", "ModelUnavailable", "default_manager",
           "register_default_adapters"]


@dataclass
class ModelState:
    name: str
    stage: str
    path: str | None
    status: str = "UNLOADED"
    truth: str = "UNLOADED"
    device: str = "cpu"
    loaded: bool = False
    message: str = ""
    load_time_s: float = 0.0
    infer_count: int = 0
    infer_time_s: float = 0.0
    last_error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_infer_ms(self) -> float:
        return (self.infer_time_s / self.infer_count * 1000.0) if self.infer_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "path": self.path,
            "status": self.status,
            "truth": self.truth,
            "device": self.device,
            "loaded": self.loaded,
            "message": self.message,
            "load_time_s": round(self.load_time_s, 4),
            "infer_count": self.infer_count,
            "avg_infer_ms": round(self.avg_infer_ms, 3),
            "last_error": self.last_error,
            "meta": self.meta,
        }


class ModelManager:
    """Registre + cache + cycle de vie des modèles."""

    def __init__(self, settings: Settings | None = None, auto_register: bool = True):
        self.settings = settings or load_settings()
        self._loaders: dict[str, Callable[[ModelSpec, "ModelManager"], Any]] = {}
        self._instances: dict[str, Any] = {}
        self._states: dict[str, ModelState] = {}
        self._lock = threading.RLock()
        if auto_register:
            register_default_adapters(self)

    # -- enregistrement ---------------------------------------------------
    def register(self, name: str, loader: Callable[[ModelSpec, "ModelManager"], Any]) -> None:
        self._loaders[name] = loader

    def has(self, name: str) -> bool:
        return name in self._loaders and name in self.settings.models

    def spec(self, name: str) -> ModelSpec:
        return self.settings.models[name]

    # -- état -------------------------------------------------------------
    def state(self, name: str) -> ModelState:
        spec = self.settings.models.get(name)
        if name not in self._states:
            self._states[name] = ModelState(
                name=name,
                stage=spec.stage if spec else "unknown",
                path=str(spec.resolved_path()) if spec and spec.resolved_path() else None,
            )
        return self._states[name]

    def statuses(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name in self.settings.models:
            out[name] = self.state(name).to_dict()
        return out

    # -- chargement -------------------------------------------------------
    def available(self, name: str) -> bool:
        """Vrai si le modèle est activé et que son checkpoint (s'il existe) est là.

        Un modèle sans fichier de poids (ex. backend novel-view) est considéré
        disponible dès lors qu'il est activé : c'est l'adapter qui tranche.
        """
        spec = self.settings.models.get(name)
        if spec is None or not spec.enabled:
            return False
        p = spec.resolved_path()
        return p is None or p.is_file()

    def load(self, name: str, force: bool = False) -> Any:
        """Charge (paresseusement) et renvoie l'instance du modèle.

        Renvoie ``None`` si le modèle est indisponible/désactivé (jamais
        d'exception pour un simple manque de checkpoint : le pipeline doit
        pouvoir continuer). Les erreurs d'exécution réelles sont propagées.
        """
        with self._lock:
            st = self.state(name)
            if name in self._instances and not force:
                return self._instances[name]
            spec = self.settings.models.get(name)
            if spec is None:
                self._set_unavailable(st, UNAVAILABLE, f"Unknown model '{name}'")
                return None
            if not spec.enabled:
                self._set_unavailable(st, DISABLED, "Disabled by configuration")
                return None
            if name not in self._loaders:
                self._set_unavailable(st, UNAVAILABLE, f"No adapter registered for '{name}'")
                return None
            path = spec.resolved_path()
            # Un chemin explicitement fourni mais absent => CHECKPOINT_MISSING.
            # Si path est None, l'adapter décide (certains backends n'ont pas de
            # fichier de poids : il lèvera ModelUnavailable si nécessaire).
            if path is not None and not path.is_file():
                st.meta["required"] = str(path)
                self._set_unavailable(st, CHECKPOINT_MISSING, "Checkpoint missing")
                return None

            loader = self._loaders[name]
            resolved_device = detect_device(spec.device or self.settings.device)
            st.device = resolved_device
            t0 = time.perf_counter()
            try:
                inst = loader(spec, self)
            except ModelUnavailable as exc:
                # runtime/dépendance/config absente ou incompatible : ce n'est
                # pas une erreur d'exécution mais une indisponibilité *classée*.
                self._set_unavailable(st, getattr(exc, "truth", UNAVAILABLE), str(exc))
                logger.warning("model '%s' unavailable (%s): %s",
                               name, st.truth, exc)
                return None
            except Exception as exc:  # erreur d'exécution réelle -> remontée
                st.status = ERROR
                st.truth = LOAD_ERROR
                st.message = f"{type(exc).__name__}: {exc}"
                st.last_error = str(exc)
                st.loaded = False
                logger.error("model '%s' load failed: %s", name, exc)
                raise
            st.load_time_s = time.perf_counter() - t0
            # Le loader peut renvoyer un tuple (instance, message) ou un dict
            if isinstance(inst, tuple) and len(inst) == 2:
                inst, message = inst
                st.message = message
            else:
                st.message = "Loaded"
            # READY == checkpoint présent + architecture compatible + chargement
            # réussi + initialisation réussie (instance non nulle).
            st.status = OK
            st.truth = READY if inst is not None else UNAVAILABLE
            st.loaded = inst is not None
            st.meta.setdefault("device", resolved_device)
            self._instances[name] = inst
            logger.info("model '%s' loaded in %.3fs on %s", name, st.load_time_s, resolved_device)
            return inst

    def _set_unavailable(self, st: ModelState, truth: str, message: str) -> None:
        """Renseigne un état d'indisponibilité *classé* (jamais un faux succès)."""
        st.truth = truth
        st.status = pipeline_status(truth)
        st.message = message
        st.loaded = False

    def get(self, name: str) -> Any:
        """Renvoie l'instance si déjà chargée, sinon tente un chargement."""
        return self.load(name)

    def unload(self, name: str) -> None:
        with self._lock:
            inst = self._instances.pop(name, None)
            if inst is not None:
                # Libère les références GPU si possible
                for attr in ("model", "net", "predictor"):
                    obj = getattr(inst, attr, None)
                    if obj is not None and hasattr(obj, "to"):
                        try:
                            obj.to("cpu")
                        except Exception:
                            pass
                del inst
            st = self.state(name)
            st.loaded = False
            st.status = "UNLOADED"
            st.truth = "UNLOADED"
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def unload_all(self) -> None:
        for name in list(self._instances):
            self.unload(name)

    # -- timing -----------------------------------------------------------
    def record_inference(self, name: str, dt: float) -> None:
        st = self.state(name)
        st.infer_count += 1
        st.infer_time_s += dt

    def time_it(self, name: str):
        """Context manager mesurant le temps d'inférence d'un modèle."""
        manager = self

        class _Timer:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                manager.record_inference(name, time.perf_counter() - self_inner.t0)
                return False

        return _Timer()

    # -- robustesse -------------------------------------------------------
    def try_stage(self, name: str) -> tuple[Any, StageResult]:
        """Tente de charger un modèle et renvoie ``(instance, StageResult)``.

        Ne lève jamais : pour un manque de checkpoint renvoie
        ``MODEL_UNAVAILABLE`` ; pour une erreur d'exécution renvoie ``ERROR``.
        Le détail honnête de l'indisponibilité est conservé dans
        ``metrics["truth"]`` (``CHECKPOINT_MISSING``, ``DEPENDENCY_MISSING``,
        ``INCOMPATIBLE``, ``GPU_REQUIRED``, ``LOAD_ERROR`` …).
        """
        spec = self.settings.models.get(name)
        try:
            inst = self.load(name)
        except Exception as exc:
            st = self.state(name)
            return None, StageResult(name, ERROR, st.message or str(exc),
                                     metrics={"device": st.device,
                                              "truth": st.truth or LOAD_ERROR})
        st = self.state(name)
        if inst is None:
            return None, StageResult(name, pipeline_status(st.truth),
                                     st.message or "Unavailable",
                                     metrics={"required": st.path,
                                              "truth": st.truth})
        return inst, StageResult(name, OK, st.message or "Loaded",
                                 evidence="VISIBLE_FROM_SOURCE",
                                 metrics={"device": st.device,
                                          "path": st.path,
                                          "truth": st.truth})

    def health(self) -> dict[str, Any]:
        """Résumé lisible de l'état de tous les modèles."""
        checks = []
        for name, spec in self.settings.models.items():
            st = self.state(name)
            exists = bool(spec.resolved_path() and spec.resolved_path().is_file())
            checks.append({
                "name": name,
                "stage": spec.stage,
                "enabled": spec.enabled,
                "checkpoint_present": exists,
                "status": st.status if st.status != "UNLOADED" else ("READY" if exists and spec.enabled else st.status),
                "truth": st.truth,
                "device": st.device,
                "avg_infer_ms": round(st.avg_infer_ms, 3),
            })
        return {
            "device": detect_device(self.settings.device),
            "cuda": _cuda_info(),
            "torch": _torch_version(),
            "models": checks,
        }

    def diagnose(self, name: str) -> dict[str, Any]:
        """Force une tentative de chargement et renvoie l'état *véridique*.

        Utile pour un rapport honnête (CLI/tests) : distingue explicitement
        ``CHECKPOINT_MISSING`` / ``DEPENDENCY_MISSING`` / ``INCOMPATIBLE`` /
        ``GPU_REQUIRED`` / ``LOAD_ERROR`` / ``READY``.
        """
        try:
            self.load(name, force=True)
        except Exception:  # une LOAD_ERROR réelle est reflétée dans l'état
            pass
        st = self.state(name)
        spec = self.settings.models.get(name)
        return {
            "name": name,
            "stage": spec.stage if spec else "unknown",
            "device": st.device,
            "truth": st.truth,
            "status": st.status,
            "loaded": st.loaded,
            "message": st.message,
            "path": st.path,
            "meta": dict(st.meta),
        }

    def readiness_report(self) -> dict[str, Any]:
        """Vérité par modèle, sans faux succès (READY seulement si réellement chargé)."""
        report: dict[str, Any] = {}
        for name in self.settings.models:
            report[name] = self.diagnose(name)
        return report


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def _cuda_info() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"available": True, "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda}
    except Exception:
        pass
    return {"available": False, "count": 0}


# ---------------------------------------------------------------------------
# Registre par défaut des adapters
# ---------------------------------------------------------------------------
def register_default_adapters(manager: "ModelManager") -> None:
    from .adapters_yolo import load_detector
    from .adapters_sam2 import load_segmenter
    from .adapters_depth import load_depth
    from .adapters_reid import load_reid
    from .adapters_rife import load_interpolation
    from .adapters_seva import load_virtual_camera
    from .adapters_sr import load_super_resolution
    from .adapters_pose import load_pose

    manager.register("detector", load_detector)
    manager.register("segmenter", load_segmenter)
    manager.register("depth", load_depth)
    manager.register("reid", load_reid)
    manager.register("interpolation", load_interpolation)
    manager.register("super_resolution", load_super_resolution)
    manager.register("virtual_camera", load_virtual_camera)
    manager.register("pose", load_pose)


_DEFAULT_MANAGER: ModelManager | None = None


def default_manager() -> ModelManager:
    """Instance partagée (pratique pour l'API/perf)."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = ModelManager()
    return _DEFAULT_MANAGER
