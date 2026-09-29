"""Adapter super-résolution (extensible).

Aucun modèle SR n'est fourni : l'adapter est désactivé par défaut et
``MODEL_UNAVAILABLE``. Un poids peut être branché via
``GOALREEL_SUPER_RESOLUTION_WEIGHTS`` (ex. Real-ESRGAN), sans changer le code.
"""
from __future__ import annotations

from ..config import ModelSpec
from .manager import ModelUnavailable


def load_super_resolution(spec: ModelSpec, manager):
    path = spec.resolved_path()
    if path is None or not path.is_file():
        raise ModelUnavailable("No super-resolution checkpoint configured")
    raise ModelUnavailable(
        "Super-resolution checkpoint present but no runtime adapter implemented "
        "(add a Real-ESRGAN/SwinIR adapter here)"
    )
