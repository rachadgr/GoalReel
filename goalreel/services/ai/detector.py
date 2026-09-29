"""Service détection (joueurs / ballon)."""
from __future__ import annotations

from typing import Any

from ...models.manager import ModelManager


class DetectorService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("detector")

    def load(self) -> Any:
        return self.manager.load("detector")

    def detect(self, frame, conf: float | None = None) -> list[dict[str, Any]]:
        model = self.manager.load("detector")
        if model is None:
            return []
        with self.manager.time_it("detector"):
            return model.detect(frame, conf)

    def players(self, frame, conf: float | None = None):
        model = self.manager.load("detector")
        if model is None:
            return []
        with self.manager.time_it("detector"):
            return model.players(frame, conf)

    def ball(self, frame, conf: float | None = None):
        model = self.manager.load("detector")
        if model is None:
            return []
        with self.manager.time_it("detector"):
            return model.ball(frame, conf)
