"""Cycle de vie et stockage des jobs de génération GoalReel Reel Studio.

Un **job** représente une génération de clip Wan 2.2. Il possède un identifiant
unique, un état parmi :class:`JobState`, une progression 0-100, un message, des
horodatages et, à la fin, une URL de résultat ou une erreur.

Le store est **thread-safe** (les générations tournent dans un thread de fond)
et fournit une conversion JSON stable via :meth:`Job.to_dict`, ce qui garantit
un format de réponse d'API constant et clair.

L'``prompt_id`` renvoyé par ComfyUI est conservé **séparément** du ``job_id`` :
on ne suppose jamais que les deux sont égaux.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobState(str, Enum):
    """États du cycle de vie d'un job."""

    QUEUED = "QUEUED"
    UPLOADING = "UPLOADING"
    SUBMITTING = "SUBMITTING"
    GENERATING = "GENERATING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """Un job de génération, sérialisable en JSON."""

    id: str
    state: JobState = JobState.QUEUED
    progress: int = 0
    message: str = "Job queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    result_url: str | None = None
    error: str | None = None
    # Paramètres demandés (utiles pour l'audit et la reproductibilité)
    request: dict[str, Any] = field(default_factory=dict)
    # Identifiants internes ComfyUI (jamais confondus avec job_id)
    prompt_id: str | None = None
    comfy_filename: str | None = None
    # Chemin local du fichier de sortie (non exposé tel quel)
    output_path: str | None = None

    def update(
        self,
        *,
        state: JobState | None = None,
        progress: int | None = None,
        message: str | None = None,
        result_url: str | None = None,
        error: str | None = None,
        prompt_id: str | None = None,
        comfy_filename: str | None = None,
        output_path: str | None = None,
    ) -> None:
        """Met à jour le job et rafraîchit ``updated_at``."""
        if state is not None:
            self.state = state
        if progress is not None:
            self.progress = max(0, min(100, int(progress)))
        if message is not None:
            self.message = message
        if result_url is not None:
            self.result_url = result_url
        if error is not None:
            self.error = error
        if prompt_id is not None:
            self.prompt_id = prompt_id
        if comfy_filename is not None:
            self.comfy_filename = comfy_filename
        if output_path is not None:
            self.output_path = output_path
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON stable et publique (pas de chemins locaux)."""
        return {
            "job_id": self.id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_url": self.result_url,
            "error": self.error,
            "request": self.request,
        }


class JobStore:
    """Store en mémoire, thread-safe, des jobs de génération."""

    def __init__(self, max_jobs: int = 500) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._max_jobs = max_jobs

    def create(self, request: dict[str, Any] | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, request=dict(request or {}))
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            # Éviction FIFO des jobs les plus anciens
            while len(self._order) > self._max_jobs:
                oldest = self._order.pop(0)
                self._jobs.pop(oldest, None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in self._order if j in self._jobs]

    def recent(self, limit: int = 20) -> list[Job]:
        jobs = self.all()
        return list(reversed(jobs[-limit:]))

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._order.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


def sleep(seconds: float) -> None:
    """Indirection de ``time.sleep`` (facilite les tests)."""
    time.sleep(seconds)
