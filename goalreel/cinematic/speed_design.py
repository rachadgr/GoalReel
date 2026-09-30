"""Cinematic Speed Design — courbes de vitesse *événement-aware*.

Objectif produit
----------------
La V1 appliquait un rythme essentiellement uniforme. La V2 construit, pour
chaque plan, une **courbe de vitesse** motivée par la phase et la preuve :

    normal → légère accélération → ralenti d'anticipation → ralenti héros → reprise

Règles de vérité
----------------
  * Le ralenti (``factor < 1``) n'est appliqué que lorsqu'un **événement héros /
    d'action réel** le soutient (phases ``HERO`` / ``ACTION`` / ``ANTICIPATION``).
    Sans événement, la vitesse reste ``NORMAL`` (1.0) : aucun ralenti « pour
    faire cinéma ».
  * Aucune interpolation neuronale n'est prétendue. Le rendu applique une
    **déformation temporelle** (``setpts`` / ré-échantillonnage) ; le mode réel est
    enregistré honnêtement dans ``fallback`` et **jamais** étiqueté « RIFE »
    sans inférence RIFE réelle.
  * Les facteurs sont bornés dans ``[SPEED_MIN, SPEED_MAX]``.

Déterminisme : la courbe ne dépend que de la phase, de la confiance et du
comportement de vitesse — aucune source d'aléa.
"""
from __future__ import annotations

from .edit_plan import (
    ACTION,
    ANTICIPATION,
    BUILD_UP,
    CELEBRATION,
    CLIMAX,
    FINAL_HERO,
    HERO,
    HOOK,
    OUTRO,
    REACTION,
    SECONDARY,
    SPEED_ANTICIPATION,
    SPEED_HERO_SLOW,
    SPEED_MAX,
    SPEED_MIN,
    SPEED_NORMAL,
    SPEED_RANGE,
    SPEED_RECOVERY,
    SPEED_SLIGHT_FAST,
    SpeedCurve,
    round_scale,
)

# Comportement de vitesse par phase (documenté). Les phases sans événement réel
# restent au temps réel (le director les bascule sur NORMAL si non soutenues).
PHASE_SPEED_BEHAVIOR: dict[str, str] = {
    HOOK: SPEED_NORMAL,
    BUILD_UP: SPEED_SLIGHT_FAST,
    ANTICIPATION: SPEED_ANTICIPATION,
    ACTION: SPEED_ANTICIPATION,
    HERO: SPEED_HERO_SLOW,
    CLIMAX: SPEED_HERO_SLOW,
    REACTION: SPEED_RECOVERY,
    CELEBRATION: SPEED_RECOVERY,
    SECONDARY: SPEED_NORMAL,
    FINAL_HERO: SPEED_RECOVERY,
    OUTRO: SPEED_NORMAL,
}

# Raisons explicites (audit).
_REASON = {
    SPEED_NORMAL: "NO_SPEED_EVIDENCE",
    SPEED_SLIGHT_FAST: "RHYTHM_BUILD_UP",
    SPEED_ANTICIPATION: "EVENT_ANTICIPATION_LINKED",
    SPEED_HERO_SLOW: "HERO_EVENT_LINKED",
    SPEED_RECOVERY: "EVENT_RECOVERY_LINKED",
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(x: float) -> float:
    return float(min(SPEED_MAX, max(SPEED_MIN, x)))


def speed_curve(phase: str, event_supported: bool, confidence: float = 0.0,
                rife_available: bool = False,
                motion_interpolation: str | None = None) -> SpeedCurve:
    """Construit la courbe de vitesse d'une phase.

    ``event_supported`` indique qu'un événement RÉEL soutient la phase. S'il est
    faux, le ralenti est **interdit** : on retombe sur ``NORMAL`` avec la raison
    ``NO_SPEED_EVIDENCE`` (aucun ralenti « décoratif »).

    ``rife_available`` / ``motion_interpolation`` précisent l'état honnête du
    back-end d'interpolation ; le champ ``fallback`` enregistre le mode réel
    utilisé par le renderer (jamais « RIFE » si RIFE n'a pas tourné).
    """
    behavior = PHASE_SPEED_BEHAVIOR.get(phase, SPEED_NORMAL)
    if not event_supported:
        behavior = SPEED_NORMAL

    lo, hi = SPEED_RANGE.get(behavior, (1.0, 1.0))
    conf = float(min(1.0, max(0.0, confidence)))

    if behavior == SPEED_NORMAL:
        start = end = 1.0
        easing = "linear"
    elif behavior == SPEED_SLIGHT_FAST:
        start, end, easing = lo, hi, "ease_in_out"
    elif behavior == SPEED_ANTICIPATION:
        # Du plus rapide (hi) vers le ralenti d'anticipation (lo).
        start, end, easing = hi, lo, "ease_in"
    elif behavior == SPEED_HERO_SLOW:
        # Plus la preuve est forte, plus le ralenti peut descendre (borné).
        depth = _lerp(hi, lo, conf)
        start, end, easing = hi, depth, "ease_in_out"
    else:  # SPEED_RECOVERY
        start, end, easing = lo, hi, "ease_out"

    fallback = None
    if behavior != SPEED_NORMAL and not rife_available:
        fallback = motion_interpolation or "TEMPORAL_RESAMPLE_NO_RIFE"

    return SpeedCurve(
        behavior=behavior,
        start_factor=_clamp(start),
        end_factor=_clamp(end),
        easing=easing,
        reason=_REASON.get(behavior, "UNKNOWN"),
        fallback=fallback,
    )


def factors_are_bounded(curve: SpeedCurve) -> bool:
    """Contrat : toute courbe reste dans ``[SPEED_MIN, SPEED_MAX]``."""
    return (SPEED_MIN - 1e-9) <= curve.start_factor <= (SPEED_MAX + 1e-9) and \
           (SPEED_MIN - 1e-9) <= curve.end_factor <= (SPEED_MAX + 1e-9)


def mean_factor(curve: SpeedCurve) -> float:
    """Facteur moyen (documentation / QA de durée)."""
    return round_scale((curve.start_factor + curve.end_factor) / 2.0)


def has_slowdown(curve: SpeedCurve) -> bool:
    """Vrai si la courbe contient un ralenti réel (``factor < 1``)."""
    return curve.start_factor < 1.0 - 1e-9 or curve.end_factor < 1.0 - 1e-9
