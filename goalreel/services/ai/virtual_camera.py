"""Service caméra virtuelle / novel-view (SEVA).

Réutilise le ``NovelViewPlanner`` existant (``goalreel/generation``) : si le
backend SEVA est configuré et disponible, il est branché ; sinon le provider
``unavailable`` d'origine est conservé (aucune vue inventée).
"""
from __future__ import annotations

from ...models.manager import ModelManager


class VirtualCameraService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("virtual_camera")

    def load(self):
        """Renvoie un provider novel-view (SEVA) ou ``None``."""
        return self.manager.load("virtual_camera")

    def generate(self, request: dict):
        backend = self.manager.load("virtual_camera")
        if backend is None:
            return {"status": "MODEL_UNAVAILABLE",
                    "reason": "Aucune caméra virtuelle configurée (GOALREEL_NOVEL_VIEW_BACKEND)"}
        with self.manager.time_it("virtual_camera"):
            return backend.generate(request)
