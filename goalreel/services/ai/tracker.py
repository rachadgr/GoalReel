"""Service suivi — ByteTrack (sans modèle, algorithmique)."""
from __future__ import annotations

from ...core.types import StageResult
from ...vision.bytetrack import ByteTrack


class TrackerService:
    """Réutilise l'implémentation ByteTrack existante (aucune réécriture)."""

    def __init__(self, high: float = 0.5, low: float = 0.1, match: float = 0.3,
                 max_lost: int = 30):
        self.tracker = ByteTrack(high=high, low=low, match=match, max_lost=max_lost)

    def update(self, detections: list[dict]):
        return self.tracker.update(detections)

    @property
    def tracks(self):
        return self.tracker.tracks

    def status(self) -> StageResult:
        return StageResult("bytetrack", "OK", "ByteTrack association active",
                           metrics={"active_tracks": len(self.tracker.tracks)})
