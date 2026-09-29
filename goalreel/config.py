"""Configuration centrale des modèles GoalReel.

Toute la configuration est résolue depuis des variables d'environnement
(préfixe ``GOALREEL_``), avec des valeurs par défaut relatives au dépôt.
Aucun chemin ni paramètre n'est codé en dur dans les modules d'inférence :
tout passe par :func:`load_settings`.

Compatible avec le style existant (``os.getenv('GOALREEL_...')``) afin de ne
pas casser la configuration déjà documentée dans ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(f"GOALREEL_{key}")
    return val if val not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(f"GOALREEL_{key}")
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_int(key: str, default: int) -> int:
    val = os.getenv(f"GOALREEL_{key}")
    try:
        return int(val) if val not in (None, "") else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(f"GOALREEL_{key}")
    try:
        return float(val) if val not in (None, "") else default
    except ValueError:
        return default


@dataclass
class ModelSpec:
    """Spécification d'un modèle configurable et activable."""

    name: str
    stage: str
    path: str | None
    enabled: bool = True
    device: str = "auto"
    config: str | None = None
    precision: str = "fp32"
    input_size: int | None = None
    batch_size: int = 1
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_path(self) -> Path | None:
        if not self.path:
            return None
        p = Path(self.path)
        return p if p.is_absolute() else (ROOT / p)


@dataclass
class Settings:
    device: str
    models_dir: Path
    checkpoints_dir: Path
    vendor_dir: Path
    cache_dir: Path
    models: dict[str, ModelSpec]
    frame_stride: int
    max_frames: int
    log_level: str

    def get(self, name: str) -> ModelSpec:
        return self.models[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "models_dir": str(self.models_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "frame_stride": self.frame_stride,
            "max_frames": self.max_frames,
            "models": {
                k: {
                    "stage": v.stage,
                    "path": v.path,
                    "resolved": str(v.resolved_path()) if v.resolved_path() else None,
                    "enabled": v.enabled,
                    "device": v.device,
                    "precision": v.precision,
                    "input_size": v.input_size,
                    "batch_size": v.batch_size,
                    "confidence": v.confidence,
                }
                for k, v in self.models.items()
            },
        }


def detect_device(preferred: str = "auto") -> str:
    """Résout le périphérique effectif (``cuda`` / ``cpu`` / ``mps``)."""
    pref = (preferred or "auto").lower()
    if pref in ("cpu", "cuda", "cuda:0", "mps"):
        if pref.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return pref
            except Exception:
                pass
            return "cpu"
        if pref == "mps":
            try:
                import torch

                return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            except Exception:
                return "cpu"
        return pref
    # auto
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Construit les :class:`Settings` depuis l'environnement."""
    models_dir = Path(_env("MODELS_DIR", str(ROOT / "models")))
    checkpoints_dir = Path(_env("CHECKPOINTS_DIR", str(ROOT / "checkpoints")))
    vendor_dir = Path(_env("VENDOR_DIR", str(ROOT / "vendor")))
    cache_dir = Path(_env("CACHE_DIR", str(ROOT / ".model_cache")))
    device = _env("DEVICE", "auto")

    def spec(name: str, stage: str, path: str | None, **kw) -> ModelSpec:
        env_path = _env(f"{name.upper()}_WEIGHTS") or _env(f"{name.upper()}_CHECKPOINT")
        return ModelSpec(
            name=name,
            stage=stage,
            path=env_path or path,
            enabled=_env_bool(f"ENABLE_{name.upper()}", kw.pop("enabled", True)),
            device=_env(f"{name.upper()}_DEVICE", kw.pop("device", device)),
            config=_env(f"{name.upper()}_CONFIG", kw.pop("config", None)),
            precision=_env(f"{name.upper()}_PRECISION", kw.pop("precision", "fp32")),
            input_size=_env_int(f"{name.upper()}_INPUT_SIZE", kw.pop("input_size", 0)) or None,
            batch_size=_env_int(f"{name.upper()}_BATCH_SIZE", kw.pop("batch_size", 1)),
            confidence=_env_float(f"{name.upper()}_CONF", kw.pop("confidence", 0.0)) or None,
            extra=kw,
        )

    models = {
        # --- détection / segmentation / suivi / reid -----------------------
        "detector": spec(
            "detector", "detection",
            _env("YOLO_WEIGHTS", "models/detection/yolov8s.pt"),
            input_size=640, confidence=0.25,
            # rétro-compatibilité : GOALREEL_YOLO_WEIGHTS (ancien nom)
            legacy_env="GOALREEL_YOLO_WEIGHTS",
        ),
        "segmenter": spec(
            "segmenter", "segmentation",
            _env("SAM2_CHECKPOINT", "models/segmentation/sam2.1_hiera_tiny.pt"),
            config=_env("SAM2_CONFIG", "sam2.1_hiera_t"),
        ),
        "depth": spec(
            "depth", "depth",
            _env("DEPTH_CHECKPOINT", "models/depth/depth_anything_v2_vits.pth"),
            input_size=518,
        ),
        "reid": spec(
            "reid", "reid",
            _env("REID_WEIGHTS", "models/reid/osnet_x1_0_imagenet.pth"),
            input_size=256,
        ),
        "interpolation": spec(
            "interpolation", "interpolation",
            _env("RIFE_CHECKPOINT", "checkpoints/RIFE/flownet.pkl"),
        ),
        "super_resolution": spec(
            "super_resolution", "super_resolution",
            _env("SUPER_RESOLUTION_WEIGHTS", None),
            enabled=_env_bool("ENABLE_SUPER_RESOLUTION", False),
            extra={"scale": _env_int("SUPER_RESOLUTION_SCALE", 2)},
        ),
        "virtual_camera": spec(
            "virtual_camera", "virtual_camera",
            _env("SEVA_WEIGHTS", None),
            enabled=_env_bool("ENABLE_VIRTUAL_CAMERA", False),
            extra={"backend": _env("NOVEL_VIEW_BACKEND", "unavailable"),
                   "model_version": _env("SEVA_MODEL_VERSION", "1.1")},
        ),
        "pose": spec(
            "pose", "pose",
            _env("POSE_WEIGHTS", None),
        ),
    }

    return Settings(
        device=device,
        models_dir=models_dir,
        checkpoints_dir=checkpoints_dir,
        vendor_dir=vendor_dir,
        cache_dir=cache_dir,
        models=models,
        frame_stride=_env_int("FRAME_STRIDE", 15),
        max_frames=_env_int("MAX_FRAMES", 0),
        log_level=_env("LOG_LEVEL", "INFO"),
    )
