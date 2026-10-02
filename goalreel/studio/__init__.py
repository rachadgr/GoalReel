"""GoalReel AI Football Reel Studio.

Package du studio de génération de clips football basé sur ComfyUI + Wan 2.2
TI2V 5B. Il est indépendant du pipeline vision « Virtual Cinema Engine »
(:mod:`goalreel`) : il ajoute la brique *image-to-video* pilotée depuis le
téléphone ou le web, sans rien casser de l'existant.

Composants :

* :mod:`goalreel.studio.studio_config` — configuration (env ``GOALREEL_*`` /
  ``COMFY_*``) ;
* :mod:`goalreel.studio.presets`       — presets football ;
* :mod:`goalreel.studio.jobs`          — cycle de vie & store des jobs ;
* :mod:`goalreel.studio.wan22`         — générateur de workflow Wan 2.2 ;
* :mod:`goalreel.studio.comfyui`       — client HTTP ComfyUI ;
* :mod:`goalreel.studio.service`       — orchestration (background) ;
* :mod:`goalreel.studio.api`           — API FastAPI.
"""
from __future__ import annotations

__version__ = "2.4.0"

__all__ = [
    "studio_config",
    "presets",
    "jobs",
    "wan22",
    "comfyui",
    "service",
    "api",
]
