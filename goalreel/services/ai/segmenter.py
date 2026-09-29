"""Service segmentation / matting (SAM 2.1)."""
from __future__ import annotations

from typing import Any

from ...models.manager import ModelManager


class SegmenterService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("segmenter")

    def load(self) -> Any:
        return self.manager.load("segmenter")

    def segment_bbox(self, frame, bbox) -> dict[str, Any] | None:
        """Segmente un objet à partir d'une boîte englobante."""
        model = self.manager.load("segmenter")
        if model is None:
            return None
        with self.manager.time_it("segmenter"):
            model.set_image(frame)
            return model.segment_from_box(bbox)

    def segment_points(self, frame, points, labels) -> dict[str, Any] | None:
        model = self.manager.load("segmenter")
        if model is None:
            return None
        with self.manager.time_it("segmenter"):
            model.set_image(frame)
            return model.segment_from_points(points, labels)
