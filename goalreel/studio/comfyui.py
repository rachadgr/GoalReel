"""Client HTTP pour l'API ComfyUI.

Responsabilités :

* ``GET /system_stats``  → disponibilité / statut de ComfyUI ;
* ``POST /upload/image`` → upload d'une image de départ ;
* ``POST /prompt``       → soumission du workflow (retourne un ``prompt_id``) ;
* ``GET /history/{id}``  → interrogation de l'état d'un prompt ;
* ``GET /view``          → téléchargement du fichier vidéo produit.

Principes :

* aucune dépendance à un nom de fichier **fixe** : le fichier de sortie est
  découvert dynamiquement dans le ``history`` ;
* ``prompt_id`` ≠ ``job_id`` (le studio ne les confond jamais) ;
* erreurs réseau / timeouts traduits en :class:`ComfyUIError` avec message
  clair et **sans stack trace** exposée ;
* le module est conçu pour être **mocké** facilement dans les tests (transport
  HTTP injectable via ``transport`` ou ``session_factory``).
"""
from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger("goalreel.studio.comfyui")


class ComfyUIError(RuntimeError):
    """Erreur générique côté ComfyUI (réseau, HTTP, format de réponse)."""


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI est injoignable (connexion refusée, DNS, timeout)."""


class ComfyUIWorkflowError(ComfyUIError):
    """Le workflow a échoué côté ComfyUI (erreur d'exécution)."""


@dataclass
class ComfyStatus:
    """Statut de disponibilité de ComfyUI."""

    available: bool
    url: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "url": self.url,
            "message": self.message,
            "details": self.details or {},
        }


class ComfyUIClient:
    """Client synchrone minimal pour l'API ComfyUI."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = retry_backoff_s
        self._sleep = sleep
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
        )

    # ------------------------------------------------------------------ utils
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Exécute une requête avec retries sur erreurs réseau/5xx."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.request(method, url, **kwargs)
                if resp.status_code >= 500:
                    raise ComfyUIError(
                        f"ComfyUI a renvoyé HTTP {resp.status_code} sur {url}"
                    )
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError) as exc:
                last_exc = exc
                logger.warning(
                    "ComfyUI unreachable (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "ComfyUI HTTP error (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
            if attempt < self.max_retries:
                self._sleep(self.retry_backoff_s * attempt)
        raise ComfyUIUnavailable(
            "ComfyUI est injoignable. Vérifiez que le serveur tourne et que "
            "COMFY_URL est correct."
        ) from last_exc

    # ------------------------------------------------------------------ statut
    def system_stats(self) -> dict[str, Any]:
        resp = self._request("GET", "/system_stats")
        if resp.status_code != 200:
            raise ComfyUIError(f"system_stats HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ComfyUIError("Réponse system_stats illisible (JSON invalide)") from exc

    def status(self) -> ComfyStatus:
        """Retourne un :class:`ComfyStatus` sans jamais lever d'exception."""
        try:
            stats = self.system_stats()
            return ComfyStatus(
                available=True,
                url=self.base_url,
                message="ComfyUI disponible",
                details=stats,
            )
        except ComfyUIError as exc:
            return ComfyStatus(
                available=False,
                url=self.base_url,
                message=str(exc),
                details=None,
            )

    # ------------------------------------------------------------------ upload
    def upload_image(
        self,
        file_path: str | Path,
        *,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Upload une image sur ``POST /upload/image``.

        Retourne la réponse JSON de ComfyUI, contenant au minimum ``name``.
        """
        path = Path(file_path)
        if not path.is_file():
            raise ComfyUIError(f"Fichier introuvable: {path.name}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as fh:
            files = {"image": (path.name, fh, content_type)}
            data = {"overwrite": "true" if overwrite else "false"}
            if subfolder:
                data["subfolder"] = subfolder
            resp = self._request("POST", "/upload/image", files=files, data=data)

        if resp.status_code not in (200, 201):
            raise ComfyUIError(
                f"Échec de l'upload image (HTTP {resp.status_code})"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ComfyUIError("Réponse upload illisible (JSON invalide)") from exc

        name = payload.get("name")
        if not name:
            raise ComfyUIError("Réponse upload sans champ 'name'")
        return payload

    # ------------------------------------------------------------------ prompt
    def submit(self, workflow: dict[str, Any], *, client_id: str | None = None) -> str:
        """Soumet un workflow à ``POST /prompt`` et retourne le ``prompt_id``."""
        body: dict[str, Any] = {"prompt": workflow}
        if client_id:
            body["client_id"] = client_id
        resp = self._request("POST", "/prompt", json=body)

        if resp.status_code == 400:
            raise ComfyUIWorkflowError(
                "Workflow rejeté par ComfyUI (validation échouée). "
                "Vérifiez les noms de modèles et les paramètres."
            )
        if resp.status_code != 200:
            raise ComfyUIError(f"Échec de soumission (HTTP {resp.status_code})")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ComfyUIError("Réponse /prompt illisible (JSON invalide)") from exc

        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            # Certaines versions renvoient {"error": {...}} avec HTTP 200
            node_errors = payload.get("node_errors")
            raise ComfyUIWorkflowError(
                "ComfyUI n'a pas renvoyé de prompt_id"
                + (f" (node_errors={node_errors})" if node_errors else "")
            )
        return prompt_id

    # ------------------------------------------------------------------ history
    def get_history(self, prompt_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/history/{prompt_id}")
        if resp.status_code != 200:
            raise ComfyUIError(f"history HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ComfyUIError("Réponse history illisible (JSON invalide)") from exc

    def poll(
        self,
        prompt_id: str,
        *,
        timeout_s: float = 1800.0,
        interval_s: float = 2.0,
        on_tick: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Attend la fin de l'exécution d'un prompt et retourne son entrée history.

        Lève :class:`ComfyUIWorkflowError` si le prompt échoue ou dépasse le
        délai imparti.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            history = self.get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                status_str = status.get("status_str", "")
                completed = status.get("completed", False)
                if completed or status_str in ("success", "error"):
                    if status_str == "error":
                        raise ComfyUIWorkflowError(
                            "ComfyUI a signalé une erreur d'exécution du workflow."
                        )
                    return entry
            if on_tick is not None:
                try:
                    on_tick(history)
                except Exception:  # pragma: no cover - callback best effort
                    pass
            if time.monotonic() >= deadline:
                raise ComfyUIWorkflowError(
                    "Délai dépassé en attendant la fin de la génération ComfyUI."
                )
            self._sleep(interval_s)

    # ------------------------------------------------------------------ output
    def view(
        self,
        filename: str,
        *,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Télécharge un fichier produit via ``GET /view``."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        resp = self._request("GET", "/view", params=params)
        if resp.status_code != 200:
            raise ComfyUIError(f"view HTTP {resp.status_code}")
        return resp.content


# ---------------------------------------------------------------------------
# Extraction du fichier de sortie depuis une entrée history
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".gif")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class OutputRef:
    """Référence d'un fichier produit par ComfyUI."""

    filename: str
    subfolder: str = ""
    folder_type: str = "output"
    kind: str = "video"  # "video" | "image"

    def to_dict(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "subfolder": self.subfolder,
            "type": self.folder_type,
            "kind": self.kind,
        }


def extract_output(history_entry: dict[str, Any]) -> OutputRef:
    """Découvre dynamiquement la vidéo (ou à défaut l'image) produite.

    Ne se base **jamais** sur un nom de fichier fixe : parcourt les ``outputs``
    de chaque nœud et choisit le premier fichier vidéo, sinon le premier fichier
    image.
    """
    outputs = history_entry.get("outputs") or {}
    video_ref: OutputRef | None = None
    image_ref: OutputRef | None = None

    for node_output in outputs.values():
        for key in ("gifs", "videos", "video"):
            for item in node_output.get(key, []) or []:
                ref = _ref_from_item(item, "video")
                if ref and video_ref is None:
                    video_ref = ref
        for key in ("images", "image"):
            for item in node_output.get(key, []) or []:
                ref = _ref_from_item(item, "image")
                if ref and image_ref is None:
                    image_ref = ref

    if video_ref is not None:
        return video_ref
    if image_ref is not None:
        return image_ref
    raise ComfyUIWorkflowError(
        "Aucun fichier de sortie trouvé dans l'historique ComfyUI."
    )


def _ref_from_item(item: Any, kind: str) -> OutputRef | None:
    if not isinstance(item, dict):
        return None
    filename = item.get("filename")
    if not filename:
        return None
    subfolder = item.get("subfolder", "") or ""
    folder_type = item.get("type", "output") or "output"
    # On ne retient que le type de fichier attendu
    lower = filename.lower()
    if kind == "video" and not lower.endswith(_VIDEO_EXTENSIONS):
        # Un fichier non-vidéo dans "gifs" reste acceptable s'il est le seul
        if not lower.endswith(_IMAGE_EXTENSIONS):
            return None
    return OutputRef(
        filename=filename, subfolder=subfolder, folder_type=folder_type, kind=kind
    )
