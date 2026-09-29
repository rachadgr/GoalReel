"""Service interpolation (RIFE) avec fallback de qualité explicite.

Si RIFE est disponible : interpolation réelle par réseau de flux optique.
Sinon : *blend* linéaire (marqué ``fallback``) — ne prétend pas être du RIFE.
"""
from __future__ import annotations

from ...models.manager import ModelManager


class InterpolationService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("interpolation")

    def interpolate(self, img0, img1, timestep: float = 0.5):
        """Renvoie ``(frame, mode)`` où ``mode`` ∈ {``rife``, ``fallback``}."""
        model = self.manager.load("interpolation")
        if model is not None:
            with self.manager.time_it("interpolation"):
                return model.interpolate(img0, img1, timestep), "rife"
        return self._blend(img0, img1, timestep), "fallback"

    @staticmethod
    def _blend(img0, img1, timestep: float):
        import cv2

        if timestep <= 0:
            return img0
        if timestep >= 1:
            return img1
        return cv2.addWeighted(img0, 1.0 - timestep, img1, timestep, 0.0)

    def slowdown_frames(self, frames, factor: int = 2):
        """Double (ou plus) la cadence en insérant des frames interpolées."""
        if factor < 2 or len(frames) < 2:
            return list(frames)
        steps = factor
        out = []
        for i in range(len(frames) - 1):
            out.append(frames[i])
            for s in range(1, steps):
                interp, _ = self.interpolate(frames[i], frames[i + 1], s / steps)
                out.append(interp)
        out.append(frames[-1])
        return out
