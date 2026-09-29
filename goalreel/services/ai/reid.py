"""Service ré-identification joueur (OSNet)."""
from __future__ import annotations

from typing import Any

from ...models.manager import ModelManager


class ReIDService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    @property
    def available(self) -> bool:
        return self.manager.available("reid")

    def embed(self, crop_bgr):
        model = self.manager.load("reid")
        if model is None:
            return None
        with self.manager.time_it("reid"):
            return model.embed(crop_bgr)

    def match(self, query_emb, candidates: dict[str, Any], threshold: float = 0.55):
        """Associe un embedding à l'identité la plus proche (cosinus)."""
        model = self.manager.load("reid")
        if model is None or query_emb is None:
            return {"status": "MODEL_UNAVAILABLE", "identity": None}
        best_id, best_sim = None, -1.0
        for identity, emb in candidates.items():
            sim = model.cosine(query_emb, emb)
            if sim > best_sim:
                best_id, best_sim = identity, sim
        ok = best_sim >= threshold
        return {"status": "OK" if ok else "UNKNOWN", "identity": best_id if ok else None,
                "similarity": best_sim}
