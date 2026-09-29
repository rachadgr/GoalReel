"""Service super-résolution — espace d'extension.

Aucun modèle de SR n'est fourni. Le service applique un upscale classique
Lanczos (``fallback``) et signale explicitement que ce n'est **pas** du SR
neuronal. Un modèle (ex. Real-ESRGAN) peut être branché plus tard via
``GOALREEL_SUPER_RESOLUTION_WEIGHTS``.
"""
from __future__ import annotations

import cv2

from ...models.manager import ModelManager


class SuperResolutionService:
    def __init__(self, manager: ModelManager, scale: int = 2):
        self.manager = manager
        self.scale = scale

    @property
    def available(self) -> bool:
        return self.manager.available("super_resolution")

    def upscale(self, frame):
        """Renvoie ``(frame, mode)`` — ``fallback`` (Lanczos) en l'absence de modèle."""
        model = self.manager.load("super_resolution")
        if model is not None:
            with self.manager.time_it("super_resolution"):
                return model.upscale(frame), "model"
        h, w = frame.shape[:2]
        out = cv2.resize(frame, (w * self.scale, h * self.scale),
                         interpolation=cv2.INTER_LANCZOS4)
        return out, "fallback"
