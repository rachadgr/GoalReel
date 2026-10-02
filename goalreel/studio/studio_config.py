"""Configuration du GoalReel AI Football Reel Studio.

Ce module centralise **toute** la configuration du studio de génération de
clips (FastAPI + ComfyUI + Wan 2.2 TI2V 5B). Il est volontairement séparé de
:mod:`goalreel.config` (qui décrit le pipeline vision « Virtual Cinema
Engine ») afin de ne rien casser dans l'architecture existante.

Résolution de la configuration, par ordre de priorité :

1. arguments explicites passés à :func:`load_studio_settings` ;
2. variables d'environnement ``GOALREEL_*`` **et** ``COMFY_*`` / ``HOST`` /
   ``PORT`` (pour rester compatible avec le ``.env.example`` demandé) ;
3. valeurs par défaut pragmatiques pour un GPU T4 (16 Go).

Aucun secret n'est stocké en dur : tout passe par l'environnement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Racine du dépôt (goalreel/studio/studio_config.py -> parents[2] == repo root)
ROOT = Path(__file__).resolve().parents[2]


def _get(env: dict[str, str], *names: str, default: str | None = None) -> str | None:
    """Retourne la première variable non vide parmi ``names``."""
    for name in names:
        val = env.get(name)
        if val is not None and val != "":
            return val
    return default


def _get_int(env: dict[str, str], *names: str, default: int) -> int:
    raw = _get(env, *names)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _get_float(env: dict[str, str], *names: str, default: float) -> float:
    raw = _get(env, *names)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _get_bool(env: dict[str, str], *names: str, default: bool) -> bool:
    raw = _get(env, *names)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _resolve_path(value: str | None, default: Path) -> Path:
    """Résout un chemin relatif par rapport à la racine du dépôt."""
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)


@dataclass(frozen=True)
class ComfyModels:
    """Noms de fichiers des modèles **déjà présents** dans ComfyUI.

    Ces noms correspondent exactement à l'environnement Wan 2.2 cible décrit
    dans le README. Ils ne sont jamais téléchargés ni versionnés par le dépôt.
    """

    diffusion: str = "wan2.2_ti2v_5B_fp16.safetensors"
    vae: str = "wan2.2_vae.safetensors"
    text_encoder: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"


@dataclass(frozen=True)
class GenerationDefaults:
    """Paramètres par défaut de génération Wan 2.2 TI2V 5B (optimisés T4)."""

    width: int = 832
    height: int = 480
    length: int = 49
    fps: float = 16.0
    steps: int = 20
    cfg: float = 5.0
    seed: int = 123456789
    shift: float = 3.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    positive_prompt: str = (
        "cinematic realistic football player in action, professional stadium, "
        "natural movement, realistic lighting, cinematic camera"
    )
    negative_prompt: str = (
        "blurry, distorted body, extra limbs, duplicate player, flickering, "
        "artifacts, low quality"
    )


@dataclass(frozen=True)
class StudioSettings:
    """Configuration complète du studio."""

    comfy_url: str
    host: str
    port: int
    max_upload_mb: int
    cors_origins: list[str]
    upload_dir: Path
    output_dir: Path
    job_timeout_s: int
    poll_interval_s: float
    max_retries: int
    retry_backoff_s: float
    http_timeout_s: float
    models: ComfyModels = field(default_factory=ComfyModels)
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)
    allow_download_proxy: bool = True

    # --- dérivés pratiques -------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Crée les répertoires de travail si nécessaire."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Vue sérialisable et **sans secret** de la configuration."""
        return {
            "comfy_url": self.comfy_url,
            "host": self.host,
            "port": self.port,
            "max_upload_mb": self.max_upload_mb,
            "cors_origins": list(self.cors_origins),
            "job_timeout_s": self.job_timeout_s,
            "poll_interval_s": self.poll_interval_s,
            "max_retries": self.max_retries,
            "models": {
                "diffusion": self.models.diffusion,
                "vae": self.models.vae,
                "text_encoder": self.models.text_encoder,
            },
            "defaults": {
                "width": self.defaults.width,
                "height": self.defaults.height,
                "length": self.defaults.length,
                "fps": self.defaults.fps,
                "steps": self.defaults.steps,
                "cfg": self.defaults.cfg,
                "seed": self.defaults.seed,
            },
        }


def load_studio_settings(env: dict[str, str] | None = None) -> StudioSettings:
    """Construit les :class:`StudioSettings` depuis l'environnement.

    Accepte ``GOALREEL_STUDIO_*``, ``COMFY_*`` ainsi que ``HOST`` / ``PORT`` /
    ``MAX_UPLOAD_MB`` / ``CORS_ORIGINS`` pour une compatibilité maximale avec
    le ``.env.example`` demandé.
    """
    env = dict(os.environ if env is None else env)

    comfy_url = (
        _get(env, "GOALREEL_STUDIO_COMFY_URL", "COMFY_URL", "COMFYUI_URL")
        or "http://127.0.0.1:8188"
    ).rstrip("/")

    host = _get(env, "GOALREEL_STUDIO_HOST", "HOST", default="0.0.0.0") or "0.0.0.0"
    port = _get_int(env, "GOALREEL_STUDIO_PORT", "PORT", default=7860)
    max_upload_mb = _get_int(
        env, "GOALREEL_STUDIO_MAX_UPLOAD_MB", "MAX_UPLOAD_MB", default=20
    )

    cors_raw = _get(env, "GOALREEL_STUDIO_CORS_ORIGINS", "CORS_ORIGINS", default="*")
    cors_origins = [o.strip() for o in (cors_raw or "*").split(",") if o.strip()]
    if not cors_origins:
        cors_origins = ["*"]

    upload_dir = _resolve_path(
        _get(env, "GOALREEL_STUDIO_UPLOAD_DIR", "UPLOAD_DIR"),
        ROOT / "outputs" / "studio" / "uploads",
    )
    output_dir = _resolve_path(
        _get(env, "GOALREEL_STUDIO_OUTPUT_DIR", "OUTPUT_DIR"),
        ROOT / "outputs" / "studio" / "reels",
    )

    models = ComfyModels(
        diffusion=_get(
            env,
            "GOALREEL_STUDIO_DIFFUSION_MODEL",
            "WAN_DIFFUSION_MODEL",
            default=ComfyModels.diffusion,
        ),
        vae=_get(env, "GOALREEL_STUDIO_VAE_MODEL", "WAN_VAE_MODEL", default=ComfyModels.vae),
        text_encoder=_get(
            env,
            "GOALREEL_STUDIO_TEXT_ENCODER",
            "WAN_TEXT_ENCODER",
            default=ComfyModels.text_encoder,
        ),
    )

    defaults = GenerationDefaults(
        width=_get_int(env, "GOALREEL_STUDIO_WIDTH", "WAN_WIDTH", default=GenerationDefaults.width),
        height=_get_int(env, "GOALREEL_STUDIO_HEIGHT", "WAN_HEIGHT", default=GenerationDefaults.height),
        length=_get_int(env, "GOALREEL_STUDIO_LENGTH", "WAN_LENGTH", default=GenerationDefaults.length),
        fps=_get_float(env, "GOALREEL_STUDIO_FPS", "WAN_FPS", default=GenerationDefaults.fps),
        steps=_get_int(env, "GOALREEL_STUDIO_STEPS", "WAN_STEPS", default=GenerationDefaults.steps),
        cfg=_get_float(env, "GOALREEL_STUDIO_CFG", "WAN_CFG", default=GenerationDefaults.cfg),
        seed=_get_int(env, "GOALREEL_STUDIO_SEED", "WAN_SEED", default=GenerationDefaults.seed),
    )

    return StudioSettings(
        comfy_url=comfy_url,
        host=host,
        port=port,
        max_upload_mb=max_upload_mb,
        cors_origins=cors_origins,
        upload_dir=upload_dir,
        output_dir=output_dir,
        job_timeout_s=_get_int(env, "GOALREEL_STUDIO_JOB_TIMEOUT", "JOB_TIMEOUT", default=1800),
        poll_interval_s=_get_float(env, "GOALREEL_STUDIO_POLL_INTERVAL", default=2.0),
        max_retries=_get_int(env, "GOALREEL_STUDIO_MAX_RETRIES", default=3),
        retry_backoff_s=_get_float(env, "GOALREEL_STUDIO_RETRY_BACKOFF", default=1.5),
        http_timeout_s=_get_float(env, "GOALREEL_STUDIO_HTTP_TIMEOUT", default=30.0),
        models=models,
        defaults=defaults,
        allow_download_proxy=_get_bool(env, "GOALREEL_STUDIO_ALLOW_DOWNLOAD_PROXY", default=True),
    )
