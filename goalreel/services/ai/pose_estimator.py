"""Service estimation de posture (optionnel)."""
from __future__ import annotations

from ...models.manager import ModelManager


class PoseService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("pose")

    def estimate(self, frame):
        model = self.manager.load("pose")
        if model is None:
            return []
        with self.manager.time_it("pose"):
            return model.estimate(frame)
