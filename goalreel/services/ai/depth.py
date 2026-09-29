"""Service profondeur (Depth Anything V2)."""
from __future__ import annotations

from ...models.manager import ModelManager


class DepthService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("depth")

    def infer(self, frame):
        model = self.manager.load("depth")
        if model is None:
            return None
        with self.manager.time_it("depth"):
            return model.infer(frame)

    def foreground_mask(self, frame, percentile: float = 40.0):
        """Masque approximatif des objets proches (pour séparation fond/sujet)."""
        depth = self.infer(frame)
        if depth is None:
            return None
        import numpy as np

        thr = np.percentile(depth, percentile)
        return (depth > thr).astype("uint8") * 255
