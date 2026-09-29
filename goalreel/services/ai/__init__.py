"""Couche d'inférence unifiée (façade au-dessus du ModelManager).

Chaque *service* encapsule une capacité métier du pipeline GoalReel et sait
utiliser le modèle sous-jacent **si disponible**, sinon appliquer un fallback
déterministe (jamais une sortie simulée illégitime).

  * ``detector``          : détection joueurs / ballon (YOLO)
  * ``segmenter``         : segmentation (SAM 2.1)
  * ``tracker``           : suivi (ByteTrack — sans modèle)
  * ``pose_estimator``    : posture (optionnel)
  * ``reid``              : ré-identification (OSNet)
  * ``depth``             : profondeur (Depth Anything V2)
  * ``interpolation``     : interpolation de frames (RIFE + fallback)
  * ``super_resolution``  : montée en résolution
  * ``virtual_camera``    : caméra virtuelle / novel view (SEVA)
"""
from .detector import DetectorService
from .segmenter import SegmenterService
from .tracker import TrackerService
from .pose_estimator import PoseService
from .reid import ReIDService
from .depth import DepthService
from .interpolation import InterpolationService
from .super_resolution import SuperResolutionService
from .virtual_camera import VirtualCameraService

__all__ = [
    "DetectorService",
    "SegmenterService",
    "TrackerService",
    "PoseService",
    "ReIDService",
    "DepthService",
    "InterpolationService",
    "SuperResolutionService",
    "VirtualCameraService",
]
