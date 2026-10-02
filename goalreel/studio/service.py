"""Service de génération GoalReel Reel Studio.

Orchestre le cycle de vie d'un job de génération :

    QUEUED → UPLOADING → SUBMITTING → GENERATING → PROCESSING → COMPLETED
                                                              ↘ FAILED

La génération s'exécute dans un **thread de fond** afin de ne jamais bloquer la
requête HTTP. Le service :

* valide et prépare l'image de départ ;
* construit le workflow Wan 2.2 via :mod:`goalreel.studio.wan22` ;
* pilote le client ComfyUI (:mod:`goalreel.studio.comfyui`) ;
* découvre dynamiquement le fichier vidéo de sortie ;
* copie le résultat dans ``output_dir`` et expose une ``result_url`` ;
* traduit toute erreur en message clair, sans fuite de stack trace.

La fabrique de client est injectable (``client_factory``) pour permettre le
mock dans les tests sans GPU ni ComfyUI réel.
"""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from . import wan22
from .comfyui import (
    ComfyUIClient,
    ComfyUIError,
    ComfyUIUnavailable,
    ComfyUIWorkflowError,
    extract_output,
)
from .jobs import Job, JobState, JobStore
from .studio_config import StudioSettings

logger = logging.getLogger("goalreel.studio.service")


class GenerationRequest:
    """Requête de génération validée (issue de l'API)."""

    def __init__(
        self,
        image_path: str,
        *,
        preset_id: str | None = None,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        length: int | None = None,
        fps: float | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.image_path = image_path
        self.preset_id = preset_id
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.length = length
        self.fps = fps
        self.steps = steps
        self.cfg = cfg
        self.seed = seed

    def to_public_dict(self) -> dict[str, Any]:
        """Paramètres publics (sans chemin local absolu)."""
        return {
            "preset_id": self.preset_id,
            "width": self.width,
            "height": self.height,
            "length": self.length,
            "fps": self.fps,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": self.seed,
            "has_positive_override": bool(self.positive_prompt),
            "has_negative_override": bool(self.negative_prompt),
        }


class GenerationService:
    """Service de génération thread-safe."""

    def __init__(
        self,
        settings: StudioSettings,
        store: JobStore | None = None,
        *,
        client_factory: Callable[[], ComfyUIClient] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or JobStore()
        self._client_factory = client_factory or self._default_client
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _default_client(self) -> ComfyUIClient:
        return ComfyUIClient(
            self.settings.comfy_url,
            timeout_s=self.settings.http_timeout_s,
            max_retries=self.settings.max_retries,
            retry_backoff_s=self.settings.retry_backoff_s,
        )

    # ------------------------------------------------------------------ status
    def comfy_status(self) -> dict[str, Any]:
        """Statut de ComfyUI (jamais bloquant, jamais d'exception)."""
        client = self._client_factory()
        try:
            return client.status().to_dict()
        finally:
            client.close()

    # ------------------------------------------------------------------ create
    def start(self, request: GenerationRequest) -> Job:
        """Crée un job et lance la génération en arrière-plan."""
        job = self.store.create(request.to_public_dict())
        thread = threading.Thread(
            target=self._run,
            args=(job, request),
            name=f"goalreel-job-{job.id}",
            daemon=True,
        )
        with self._lock:
            self._threads[job.id] = thread
        thread.start()
        return job

    def run_sync(self, job: Job, request: GenerationRequest) -> Job:
        """Exécute la génération de façon synchrone (utile aux tests)."""
        self._run(job, request)
        return job

    # ------------------------------------------------------------------ worker
    def _run(self, job: Job, request: GenerationRequest) -> None:
        try:
            self._generate(job, request)
        except ComfyUIUnavailable as exc:
            self._fail(job, f"ComfyUI injoignable: {exc}")
        except ComfyUIWorkflowError as exc:
            self._fail(job, f"Échec du workflow ComfyUI: {exc}")
        except ComfyUIError as exc:
            self._fail(job, f"Erreur ComfyUI: {exc}")
        except Exception as exc:  # noqa: BLE001 - filet de sécurité
            logger.exception("Unexpected error for job %s", job.id)
            self._fail(job, f"Erreur interne inattendue: {type(exc).__name__}")

    def _fail(self, job: Job, message: str) -> None:
        job.update(state=JobState.FAILED, message=message, error=message, progress=100)
        logger.error("Job %s failed: %s", job.id, message)

    def _generate(self, job: Job, request: GenerationRequest) -> None:
        settings = self.settings
        settings.ensure_dirs()

        client = self._client_factory()
        try:
            # --- 1. UPLOADING ------------------------------------------------
            job.update(
                state=JobState.UPLOADING,
                progress=10,
                message="Envoi de l'image à ComfyUI…",
            )
            uploaded = client.upload_image(request.image_path)
            comfy_image_name = uploaded.get("name")
            job.update(comfy_filename=comfy_image_name, progress=20)

            # --- 2. SUBMITTING ----------------------------------------------
            job.update(
                state=JobState.SUBMITTING,
                progress=25,
                message="Soumission du workflow Wan 2.2…",
            )
            params = wan22.default_params(
                image_name=comfy_image_name,
                preset_id=request.preset_id,
                positive_override=request.positive_prompt,
                negative_override=request.negative_prompt,
                width=request.width,
                height=request.height,
                length=request.length,
                fps=request.fps,
                steps=request.steps,
                cfg=request.cfg,
                seed=request.seed,
                models=settings.models,
                filename_prefix=f"GoalReel/{job.id}",
            )
            workflow = wan22.build_workflow(params)
            errors = wan22.validate_workflow(workflow)
            if errors:
                raise ComfyUIWorkflowError(
                    "Workflow invalide: " + "; ".join(errors)
                )

            prompt_id = client.submit(workflow)
            # prompt_id ComfyUI ≠ job_id : conservés séparément.
            job.update(prompt_id=prompt_id, progress=30)

            # --- 3. GENERATING ----------------------------------------------
            job.update(
                state=JobState.GENERATING,
                progress=35,
                message="Génération en cours (Wan 2.2 TI2V 5B)…",
            )

            def _on_tick(_history: dict[str, Any]) -> None:
                # Progression prudente pendant l'attente (35 → 85).
                if job.progress < 85:
                    job.update(progress=min(85, job.progress + 3))

            entry = client.poll(
                prompt_id,
                timeout_s=settings.job_timeout_s,
                interval_s=settings.poll_interval_s,
                on_tick=_on_tick,
            )

            # --- 4. PROCESSING ----------------------------------------------
            job.update(
                state=JobState.PROCESSING,
                progress=88,
                message="Récupération de la vidéo produite…",
            )
            out_ref = extract_output(entry)
            content = client.view(
                out_ref.filename,
                subfolder=out_ref.subfolder,
                folder_type=out_ref.folder_type,
            )

            out_path = self._store_output(job, out_ref.filename, content)
            result_url = f"/api/result/{job.id}"

            job.update(
                state=JobState.COMPLETED,
                progress=100,
                message="Vidéo prête.",
                result_url=result_url,
                output_path=str(out_path),
            )
            logger.info("Job %s completed → %s", job.id, out_path.name)
        finally:
            client.close()

    def _store_output(self, job: Job, filename: str, content: bytes) -> Path:
        """Sauvegarde le fichier de sortie sous un nom sûr (lié au job)."""
        suffix = Path(filename).suffix or ".mp4"
        safe_name = f"{job.id}{suffix}"
        out_path = self.settings.output_dir / safe_name
        out_path.write_bytes(content)
        job.update(comfy_filename=filename)
        return out_path

    # ------------------------------------------------------------------ output
    def output_path(self, job: Job) -> Path | None:
        """Chemin local du fichier de résultat (vérifié), ou ``None``."""
        if job.state is not JobState.COMPLETED or not job.output_path:
            return None
        path = Path(job.output_path)
        return path if path.is_file() else None


def copy_upload(src: str | Path, dest_dir: Path, filename: str) -> Path:
    """Copie un upload vers ``dest_dir`` sous un nom déjà sanitisé."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copyfile(src, dest)
    return dest
