"""Vocabulaire d'états *véridiques* des modèles GoalReel.

Politique de vérité : on distingue rigoureusement chaque situation réelle afin
de ne jamais présenter un modèle comme fonctionnel alors qu'il ne l'est pas.

États canoniques (``Truth``) :

  * ``READY``              — checkpoint présent + architecture compatible +
                             chargement réussi + initialisation réussie ;
  * ``CHECKPOINT_MISSING`` — aucun fichier de poids là où il est requis ;
  * ``DEPENDENCY_MISSING`` — runtime/dépendance absente (backend non installé) ;
  * ``INCOMPATIBLE``       — checkpoint présent mais architecture/config
                             incompatible avec l'adapter ;
  * ``GPU_REQUIRED``       — le modèle exige un GPU indisponible ;
  * ``LOAD_ERROR``         — erreur d'exécution réelle pendant le chargement ;
  * ``MODEL_UNAVAILABLE``  — indisponibilité non classée ;
  * ``DISABLED``           — désactivé par la configuration.

Ces états sont mappables vers les états *pipeline* existants
(``OK`` / ``MODEL_UNAVAILABLE`` / ``ERROR``) afin de **ne pas casser** l'API,
le frontend, les sorties JSON ni les tests existants.

Étapes de cycle de vie (pour des rapports honnêtes, jamais de faux succès) :
``CODE_INTEGRATED`` → ``CHECKPOINT_PRESENT`` → ``MODEL_LOADS`` →
``INFERENCE_WORKS`` → ``GPU_VALIDATED`` → ``END_TO_END_VALIDATED``.
"""
from __future__ import annotations

from typing import Final, Literal

Truth = Literal[
    "READY",
    "MODEL_UNAVAILABLE",
    "CHECKPOINT_MISSING",
    "DEPENDENCY_MISSING",
    "INCOMPATIBLE",
    "GPU_REQUIRED",
    "LOAD_ERROR",
    "DISABLED",
]

# États *pipeline* autorisés par ``goalreel.core.types.Status``.
PipelineStatus = Literal["OK", "MODEL_UNAVAILABLE", "UNKNOWN", "ERROR", "QC_REJECTED"]

# Correspondance état véridique -> état pipeline (rétro-compatibilité stricte).
_TRUTH_TO_PIPELINE: Final[dict[str, PipelineStatus]] = {
    "READY": "OK",
    "MODEL_UNAVAILABLE": "MODEL_UNAVAILABLE",
    "CHECKPOINT_MISSING": "MODEL_UNAVAILABLE",
    "DEPENDENCY_MISSING": "MODEL_UNAVAILABLE",
    "INCOMPATIBLE": "MODEL_UNAVAILABLE",
    "GPU_REQUIRED": "MODEL_UNAVAILABLE",
    "LOAD_ERROR": "ERROR",
    "DISABLED": "MODEL_UNAVAILABLE",
}

# États véridiques considérés comme « disponible/prêt ».
READY_TRUTHS: Final[frozenset[str]] = frozenset({"READY"})

# États qui signifient « ce n'est pas une erreur, juste indisponible ».
UNAVAILABLE_TRUTHS: Final[frozenset[str]] = frozenset({
    "MODEL_UNAVAILABLE",
    "CHECKPOINT_MISSING",
    "DEPENDENCY_MISSING",
    "INCOMPATIBLE",
    "GPU_REQUIRED",
    "DISABLED",
})


def pipeline_status(truth: str) -> PipelineStatus:
    """Mappe un état véridique vers l'état pipeline historique."""
    return _TRUTH_TO_PIPELINE.get(truth, "MODEL_UNAVAILABLE")


class ModelUnavailable(RuntimeError):
    """Levée par un adapter lorsqu'un modèle n'est pas exploitable.

    Porte un ``truth`` explicite (``CHECKPOINT_MISSING``, ``DEPENDENCY_MISSING``,
    ``INCOMPATIBLE``, ``GPU_REQUIRED`` …) pour que le ``ModelManager`` puisse
    rapporter l'état réel sans jamais inventer un succès.
    """

    def __init__(self, message: str, truth: str = "MODEL_UNAVAILABLE"):
        super().__init__(message)
        self.truth: str = truth if truth in _TRUTH_TO_PIPELINE else "MODEL_UNAVAILABLE"
