"""API FastAPI du GoalReel AI Football Reel Studio.

Expose le pipeline complet :

    Phone/Web UI → FastAPI → ComfyUI API → Wan 2.2 TI2V 5B → MP4 → Preview/Download

Endpoints principaux (préfixe ``/api``) :

* ``GET  /api/health``          : santé du service ;
* ``GET  /api/status``          : statut ComfyUI (available / unavailable) ;
* ``GET  /api/config``          : configuration publique (sans secret) ;
* ``GET  /api/presets``         : presets football disponibles ;
* ``POST /api/upload``          : upload d'une image de départ (validée) ;
* ``POST /api/generate``        : démarre une génération (asynchrone) ;
* ``GET  /api/status/{job_id}`` : progression d'un job ;
* ``GET  /api/jobs``            : jobs récents ;
* ``GET  /api/result/{job_id}`` : téléchargement / streaming de la vidéo.

Alias de compatibilité (sans préfixe) : ``/health``, ``/status``,
``/upload``, ``/generate``, ``/result/{job_id}``. Les routes existantes du
``server.py`` d'origine (``/health``, ``/models``, ``/pipeline``) ne sont pas
touchées : ce studio est un service distinct.

Sécurité :

* taille d'upload limitée (``MAX_UPLOAD_MB``) ;
* validation MIME + contenu réel (Pillow) ;
* noms de fichiers sanitisés, jamais de chemin arbitraire ;
* aucune stack trace renvoyée au client ;
* CORS configurable via ``CORS_ORIGINS``.
"""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import presets as presets_mod
from .jobs import JobState
from .service import GenerationRequest, GenerationService
from .studio_config import StudioSettings, load_studio_settings
from .wan22 import validate_workflow
from . import wan22

logger = logging.getLogger("goalreel.studio.api")

ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Neutralise un nom de fichier (anti path-traversal)."""
    name = Path(name or "").name  # supprime tout composant de répertoire
    name = _SAFE_NAME_RE.sub("_", name).strip("._")
    return name or "upload"


def _validate_image_bytes(content: bytes, declared_type: str | None) -> str:
    """Valide qu'un contenu est bien une image et retourne son extension.

    Lève :class:`ValueError` si le contenu n'est pas une image valide.
    """
    if not content:
        raise ValueError("Fichier vide.")
    if declared_type and declared_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            f"Type MIME non supporté: {declared_type}. "
            "Formats acceptés: PNG, JPEG, WEBP."
        )
    # Vérification du contenu réel (magic bytes) via Pillow.
    try:
        from PIL import Image

        with Image.open(BytesIO(content)) as img:
            img.verify()
        with Image.open(BytesIO(content)) as img:
            fmt = (img.format or "").lower()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Le fichier n'est pas une image valide.") from exc

    fmt_map = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "webp": ".webp"}
    if fmt not in fmt_map:
        raise ValueError(f"Format d'image non supporté: {fmt or 'inconnu'}")
    return fmt_map[fmt]


def _public_status(service: GenerationService) -> dict:
    """Statut combiné service + ComfyUI (jamais bloquant)."""
    comfy = service.comfy_status()
    return {
        "service": "goalreel-reel-studio",
        "status": "ok",
        "comfy": comfy,
        "comfy_url": service.settings.comfy_url,
    }


def create_app(settings: StudioSettings | None = None) -> FastAPI:
    """Fabrique l'application FastAPI (injectable pour les tests)."""
    settings = settings or load_studio_settings()
    settings.ensure_dirs()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Validation de référence du workflow (détection précoce de régression).
        try:
            params = wan22.default_params(image_name="example.png")
            errors = validate_workflow(wan22.build_workflow(params))
            if errors:
                logger.error("Wan 2.2 workflow validation failed: %s", errors)
            else:
                logger.info("Wan 2.2 TI2V 5B workflow validated.")
        except Exception:  # pragma: no cover
            logger.exception("Workflow validation error")
        status = service.comfy_status()
        if status["available"]:
            logger.info("ComfyUI available at %s", settings.comfy_url)
        else:
            logger.warning(
                "ComfyUI unavailable at %s: %s", settings.comfy_url, status["message"]
            )
        yield

    app = FastAPI(
        title="GoalReel — AI Football Reel Studio",
        version="2.4.0",
        description=(
            "Génération de clips football via ComfyUI + Wan 2.2 TI2V 5B. "
            "L'application reste opérationnelle même si ComfyUI est hors ligne."
        ),
        lifespan=_lifespan,
    )

    origins = settings.cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = GenerationService(settings)
    app.state.settings = settings
    app.state.service = service

    # ------------------------------------------------------------------ health
    def _health_payload() -> dict:
        return {"status": "ok", "service": "goalreel-reel-studio", "version": "2.4.0"}

    @app.get("/api/health")
    def api_health() -> dict:
        return _health_payload()

    @app.get("/health")
    def health_compat() -> dict:
        # Compatibilité : le server.py d'origine exposait déjà /health.
        return _health_payload()

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def api_status() -> dict:
        return _public_status(service)

    @app.get("/status")
    def status_compat() -> dict:
        return _public_status(service)

    @app.get("/api/config")
    def api_config() -> dict:
        return settings.public_dict()

    # ------------------------------------------------------------------ presets
    @app.get("/api/presets")
    def api_presets() -> dict:
        return {
            "default": presets_mod.DEFAULT_PRESET_ID,
            "presets": [p.to_dict() for p in presets_mod.list_presets()],
        }

    # ------------------------------------------------------------------ upload
    def _save_upload(upload: UploadFile) -> dict:
        declared = upload.content_type
        max_bytes = settings.max_upload_bytes
        content = upload.file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux (max {settings.max_upload_mb} Mo).",
            )
        try:
            ext = _validate_image_bytes(content, declared)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        upload_id = uuid.uuid4().hex[:12]
        safe_name = f"{upload_id}{ext}"
        dest = settings.upload_dir / safe_name
        dest.write_bytes(content)
        return {
            "upload_id": upload_id,
            "filename": safe_name,
            "original_filename": sanitize_filename(upload.filename or "upload"),
            "size_bytes": len(content),
            "content_type": declared,
        }

    @app.post("/api/upload")
    def api_upload(image: UploadFile = File(...)) -> dict:
        result = _save_upload(image)
        result["status"] = "ok"
        return result

    @app.post("/upload")
    def upload_compat(image: UploadFile = File(...)) -> dict:
        result = _save_upload(image)
        result["status"] = "ok"
        return result

    # ------------------------------------------------------------------ generate
    def _resolve_image_path(upload_id: str | None, filename: str | None) -> Path:
        """Résout l'image de départ **exclusivement** dans le dossier d'upload."""
        candidate: str | None = None
        if upload_id:
            candidate = upload_id
        elif filename:
            candidate = sanitize_filename(filename)
        if not candidate:
            raise HTTPException(
                status_code=400,
                detail="Fournir 'upload_id' ou 'filename' de l'image de départ.",
            )

        # Empêche tout chemin arbitraire : on ne garde que le nom de base.
        base = Path(candidate).name
        path = (settings.upload_dir / base).resolve()
        if settings.upload_dir.resolve() not in path.parents:
            raise HTTPException(status_code=400, detail="Chemin d'image invalide.")
        if not path.is_file():
            # Tentative avec extensions connues si l'id est fourni sans extension.
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                alt = (settings.upload_dir / f"{base}{ext}").resolve()
                if alt.is_file():
                    return alt
            raise HTTPException(status_code=404, detail="Image introuvable.")
        return path

    @app.post("/api/generate")
    def api_generate(
        request: Request,
        upload_id: str | None = Form(default=None),
        filename: str | None = Form(default=None),
        preset: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        negative_prompt: str | None = Form(default=None),
        width: int | None = Form(default=None),
        height: int | None = Form(default=None),
        length: int | None = Form(default=None),
        fps: float | None = Form(default=None),
        steps: int | None = Form(default=None),
        cfg: float | None = Form(default=None),
        seed: int | None = Form(default=None),
    ) -> dict:
        image_path = _resolve_image_path(upload_id, filename)
        gen_request = GenerationRequest(
            image_path=str(image_path),
            preset_id=preset,
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )
        job = service.start(gen_request)
        return {
            "status": "ok",
            "job_id": job.id,
            "state": job.state.value,
            "message": job.message,
            "status_url": f"/api/status/{job.id}",
            "result_url": f"/api/result/{job.id}",
        }

    @app.post("/generate")
    def generate_compat(
        request: Request,
        upload_id: str | None = Form(default=None),
        filename: str | None = Form(default=None),
        preset: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        negative_prompt: str | None = Form(default=None),
        width: int | None = Form(default=None),
        height: int | None = Form(default=None),
        length: int | None = Form(default=None),
        fps: float | None = Form(default=None),
        steps: int | None = Form(default=None),
        cfg: float | None = Form(default=None),
        seed: int | None = Form(default=None),
    ) -> dict:
        return api_generate(
            request, upload_id, filename, preset, prompt, negative_prompt,
            width, height, length, fps, steps, cfg, seed,
        )

    # ------------------------------------------------------------------ job
    def _job_or_404(job_id: str):
        job = service.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job introuvable.")
        return job

    @app.get("/api/status/{job_id}")
    def api_job_status(job_id: str) -> dict:
        return _job_or_404(job_id).to_dict()

    @app.get("/status/{job_id}")
    def job_status_compat(job_id: str) -> dict:
        return _job_or_404(job_id).to_dict()

    @app.get("/api/jobs")
    def api_jobs(limit: int = 20) -> dict:
        limit = max(1, min(100, limit))
        return {"jobs": [j.to_dict() for j in service.store.recent(limit)]}

    # ------------------------------------------------------------------ result
    def _serve_result(job_id: str) -> FileResponse:
        job = _job_or_404(job_id)
        if job.state is not JobState.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Résultat indisponible (état actuel: {job.state.value}).",
            )
        path = service.output_path(job)
        if path is None:
            raise HTTPException(status_code=404, detail="Fichier de résultat introuvable.")
        media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    @app.get("/api/result/{job_id}")
    def api_result(job_id: str) -> FileResponse:
        return _serve_result(job_id)

    @app.get("/result/{job_id}")
    def result_compat(job_id: str) -> FileResponse:
        return _serve_result(job_id)

    # ------------------------------------------------------- static frontend
    # Si le frontend a été buildé (`web/dist`), on le sert à la racine.
    # Les routes API ci-dessus restent prioritaires (déclarées avant le mount).
    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.is_dir():
        try:
            from fastapi.staticfiles import StaticFiles

            app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
            logger.info("Serving frontend from %s", web_dist)
        except Exception:  # pragma: no cover
            logger.warning("Frontend dist found but could not be mounted")

    # ------------------------------------------------------------------ errors
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # pragma: no cover
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "message": "Erreur interne du serveur."},
        )

    return app
