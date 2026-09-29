"""RIFE (interpolation) — façade cinematic branchée sur le ModelManager.

Conserve l'API historique ``RIFE(checkpoint).load()`` mais délègue désormais
le chargement réel au ModelManager central. ``interpolate`` applique RIFE si
le checkpoint est présent, sinon un fallback de blend (marqué explicitement).
"""
from pathlib import Path

from ..core.types import StageResult
from ..models.manager import ModelManager
from ..services.ai import InterpolationService


class RIFE:
    def __init__(self, checkpoint=None, manager=None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.manager = manager or ModelManager()
        # Permet d'utiliser un checkpoint explicite sans reconfigurer l'env.
        if checkpoint:
            self.manager.settings.models["interpolation"].path = str(self.checkpoint)

    def load(self):
        inst, st = self.manager.try_stage("interpolation")
        if inst is None:
            return StageResult("rife", st.status, st.message,
                               metrics={"required": st.metrics.get("required",
                                                                     "checkpoints/RIFE/flownet.pkl")})
        return StageResult("rife", "OK", st.message,
                           metrics={"device": self.manager.state("interpolation").device})

    def interpolate(self, img0, img1, timestep: float = 0.5):
        svc = InterpolationService(self.manager)
        frame, mode = svc.interpolate(img0, img1, timestep)
        return frame
