"""Transitions cinématiques — restreintes et motivées par les frontières d'événement.

Politique produit
-----------------
  * **coupe franche** (``HARD_CUT``) : valeur par défaut, y compris à chaque
    frontière de plan RÉELLE mesurée dans la source ;
  * **fondu court** (``DISSOLVE``) : uniquement entre deux plans d'une même
    phase lente et continue (anticipation → héros), là où une coupe serait
    abrupte — jamais comme effet décoratif ;
  * **fondus au noir** (``FADE_FROM_BLACK`` / ``FADE_TO_BLACK``) : réservés à
    l'entrée (HOOK) et à la sortie (OUTRO) de la vignette.

Interdits (aucune valeur produite par ce module) : wipes aléatoires, presets
« flashy », transitions glitch, excès de transitions.

Déterminisme : la fonction ne dépend que des phases bornantes, de la distance
temporelle et de la liste réelle des coupes de la source.
"""
from __future__ import annotations

from .edit_plan import (
    CUT_DISSOLVE,
    CUT_FADE_FROM_BLACK,
    CUT_FADE_TO_BLACK,
    CUT_HARD,
    HOOK,
    OUTRO,
    HERO,
    ANTICIPATION,
)

# Distance temporelle (frames) sous laquelle une coupe est considérée
# « adjacente » à une frontière de plan réelle (même respiration).
CUT_ADJACENCY = 3

# Phases entre lesquelles un fondu court est *justifié* (même intention, rythme
# continu) : anticipation → héros.
DISSOLVE_PAIRS = frozenset({
    (ANTICIPATION, HERO),
})


def resolve_transitions(prev_phase: str | None, phase: str, start_frame: int,
                        is_first: bool, is_last: bool,
                        real_cuts: list[int] | None = None) -> tuple[str, str, str]:
    """Renvoie ``(transition_in, transition_out, reason)`` pour un plan.

    ``transition_out`` est rempli par l'appelant sur le plan suivant ; ici on
    renvoie la transition *d'entrée* et la transition *de sortie* par défaut,
    cohérentes avec le type final de plan.
    """
    real_cuts = list(real_cuts or [])
    on_real_cut = any(abs(int(c) - int(start_frame)) <= CUT_ADJACENCY
                      for c in real_cuts)

    if is_first:
        return CUT_FADE_FROM_BLACK, CUT_HARD, "OPENING_FADE_FROM_BLACK"
    if is_last:
        return CUT_HARD, CUT_FADE_TO_BLACK, "CLOSING_FADE_TO_BLACK"
    if on_real_cut:
        return CUT_HARD, CUT_HARD, "REAL_SOURCE_CUT_BOUNDARY"
    if prev_phase is not None and (prev_phase, phase) in DISSOLVE_PAIRS:
        return CUT_DISSOLVE, CUT_HARD, "MOTIVATED_SHORT_DISSOLVE"
    return CUT_HARD, CUT_HARD, "DEFAULT_HARD_CUT"


def is_supported_transition(name: str) -> bool:
    """Contrat : seules les transitions de la liste blanche sont autorisées."""
    return name in (CUT_HARD, CUT_DISSOLVE, CUT_FADE_FROM_BLACK, CUT_FADE_TO_BLACK)


def is_outro_phase(phase: str) -> bool:
    return phase == OUTRO
