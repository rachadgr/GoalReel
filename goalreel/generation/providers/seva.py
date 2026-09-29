"""Provider novel-view réel — Stable Virtual Camera (SEVA).

Étend ``NovelViewProvider`` (API existante ``status``/``generate``) et
délègue au backend SEVA géré par le ``ModelManager``. Si SEVA n'est pas
configuré/disponible, ``status`` renvoie ``MODEL_UNAVAILABLE`` (jamais de vue
inventée).
"""
from __future__ import annotations

from .base import NovelViewProvider


class SevaProvider(NovelViewProvider):
    name = "seva"

    def __init__(self, manager=None):
        self._manager = manager

    def _backend(self):
        if self._manager is None:
            from ...models.manager import default_manager

            self._manager = default_manager()
        return self._manager.load("virtual_camera")

    def status(self):
        backend = self._backend()
        if backend is None:
            st = self._manager.state("virtual_camera") if self._manager else None
            return {"status": "MODEL_UNAVAILABLE", "provider": self.name,
                    "reason": st.message if st else "not configured"}
        return backend.status()

    def generate(self, request):
        backend = self._backend()
        if backend is None:
            return {"status": "MODEL_UNAVAILABLE",
                    "reason": "SEVA backend unavailable"}
        return backend.generate(request)
