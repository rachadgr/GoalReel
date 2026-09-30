"""Sélection du « moment héros » — intelligence football V2, guidée par la PREUVE.

Politique ``NO_EVENT_WITHOUT_EVIDENCE`` : le héros est choisi **uniquement** à
partir d'événements réellement inférés (issus de trajectoires réellement
suivies). Aucun ballon, but, identité ou événement football n'est inventé.

Problème corrigé par V2
-----------------------
La V1 classait les événements principalement via la ``confidence`` de
l'événement, dont la formule ``min(0.99, displacement / 300)`` **sature** : sur
la vidéo réelle, 16 des 50 événements valent ``0.99``, ce qui crée de
nombreuses égalités et un classement peu explicable. V2 remplace ce critère par
un **score multi-preuves pondéré et explicable**, calculé à partir d'éléments
strictement réels et mesurables :

  * ``amplitude``            — déplacement net réel du sujet (px, preuve de suivi) ;
  * ``persistence``          — nombre réel de frames où la trajectoire existe ;
  * ``continuity``           — ``frames_vues / étendue`` (pénalise les trous) ;
  * ``event_proximity``      — proximité temporelle au pic d'action réel ;
  * ``spatial_relevance``    — position verticale réelle (premier plan/terrain) ;
  * ``event_confidence``     — confiance bornée, **poids faible** (anti-saturation) ;
  * ``ball_proximity``       — proximité spatiale au ballon, **uniquement** quand
    une détection de ballon réelle existe (``sports ball`` COCO). Sinon le poids
    est redistribué (jamais de ballon inventé).

Vérité sur les classes : YOLOv8s est un modèle **COCO générique** (``person``,
``sports ball``). Il ne fournit **ni** ``referee`` **ni** ``goal`` ; ces classes
ne sont jamais prétendues. On distingue donc explicitement :

    * *personne détectée*  → ``class_name == 'person'`` ;
    * *ballon détecté*     → ``class_name == 'sports ball'`` ;
    * *événement de contexte football* → inféré des trajectoires suivies.

Déterminisme
------------
Pour une preuve identique, la sélection est **toujours** la même : aucune
dépendance à l'ordre des dictionnaires/listes. Le classement utilise une clé
totale et stable, y compris des départages explicites finaux
(``amplitude`` puis ``event_id`` décroissant).

Contrat événement → héros → caméra
-----------------------------------
``hero_moment.hero_track`` doit égaler ``camera_reframe.followed_track``. Le
``track_id`` renvoyé ici pilote directement le recadrage cinématique ; le
fallback documenté (``LONGEST_TRACK``) est appliqué par
:func:`goalreel.cinematic.reframe.build_reframe_targets` lorsque la trajectoire
du héros est indisponible, et la raison est enregistrée.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Poids du score multi-preuves (documentés, somme = 1 sans preuve ballon).
# ---------------------------------------------------------------------------
BASE_WEIGHTS = {
    "amplitude": 0.30,          # déplacement net réel (px)
    "persistence": 0.26,        # frames réellement suivies
    "continuity": 0.08,         # frames/étendue (trous pénalisés)
    "event_proximity": 0.12,    # proximité temporelle au pic d'action réel
    "spatial_relevance": 0.12,  # position verticale réelle (premier plan)
    "event_confidence": 0.12,   # confiance bornée — poids faible (anti-saturation)
}
# Ajouté (et renormalisé) UNIQUEMENT si une preuve ballon réelle existe.
BALL_WEIGHT = 0.15

# Échelles de saturation (constantes, documentées). Une valeur ``x`` est mappée
# par ``x / (x + scale)`` dans ``[0, 1)`` : croissance monotone, sans saturation
# brutale à 0.99.
DISP_SCALE = 300.0     # px : déplacement net
PERS_SCALE = 120.0     # frames : persistance
PROX_SCALE = 250.0     # frames : proximité temporelle
BALL_SCALE = 140.0     # px : proximité spatiale au ballon

# Classes COCO réellement exploitables (jamais inventées).
PERSON_CLASS = "person"
BALL_CLASS = "sports ball"


# ---------------------------------------------------------------------------
# Extraction d'évidence (surface publique conservée)
# ---------------------------------------------------------------------------
def _track_id(event):
    for tag in event.get("evidence", []):
        if isinstance(tag, str) and tag.startswith("track:"):
            try:
                return int(tag.split(":", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _displacement(event):
    for tag in event.get("evidence", []):
        if isinstance(tag, str) and tag.startswith("displacement:"):
            try:
                return float(tag.split(":", 1)[1])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def _frame_of(item, fallback_index):
    if isinstance(item, dict) and item.get("frame") is not None:
        try:
            return int(item["frame"])
        except (TypeError, ValueError):
            return fallback_index
    return fallback_index


def _bbox_of(item):
    if isinstance(item, dict):
        return item.get("bbox")
    return item


# ---------------------------------------------------------------------------
# Statistiques factuelles par trajectoire (persistance + géométrie réelle)
# ---------------------------------------------------------------------------
def track_stats(tracks):
    """Statistiques *factuelles* par trajectoire (aucune valeur inventée).

    ``tracks`` : ``{track_id: [ {frame,bbox}, ... ]}``. On renvoie pour chaque
    trajectoire : ``len`` (frames réellement suivies), ``displacement`` (distance
    réelle du centre entre la première et la dernière frame), ``span`` (étendue
    temporelle), ``continuity`` (``len/span``), ``mean_x``/``mean_y`` (centre
    moyen réel) et les bornes ``x0,y0,x1,y1`` du centre.
    """
    stats = {}
    for tid, frames in (tracks or {}).items():
        frames = frames or []
        items = [(f, _frame_of(f, i)) for i, f in enumerate(frames)]
        boxes = [b for b in (_bbox_of(f) for f, _ in items) if b is not None]
        length = len(frames)
        if not boxes:
            stats[tid] = {"len": length, "displacement": 0.0, "span": length,
                          "continuity": 0.0, "mean_x": 0.0, "mean_y": 0.0,
                          "x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0,
                          "first_frame": None, "last_frame": None}
            continue
        xs = [(float(b[0]) + float(b[2])) / 2.0 for b in boxes]
        ys = [(float(b[1]) + float(b[3])) / 2.0 for b in boxes]
        disp = ((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2) ** 0.5
        fr = sorted(f for _, f in items)
        first_f, last_f = (fr[0], fr[-1]) if fr else (None, None)
        span = (last_f - first_f + 1) if (first_f is not None) else length
        continuity = (length / span) if span else 0.0
        stats[tid] = {
            "len": length,
            "displacement": float(disp),
            "span": int(span),
            "continuity": float(min(1.0, continuity)),
            "mean_x": float(sum(xs) / len(xs)),
            "mean_y": float(sum(ys) / len(ys)),
            "x0": float(min(xs)), "x1": float(max(xs)),
            "y0": float(min(ys)), "y1": float(max(ys)),
            "first_frame": int(first_f) if first_f is not None else None,
            "last_frame": int(last_f) if last_f is not None else None,
        }
    return stats


# ---------------------------------------------------------------------------
# Helpers de score
# ---------------------------------------------------------------------------
def _sat(value, scale):
    """Saturation douce monotone dans ``[0, 1)`` : ``x / (x + scale)``."""
    value = float(value or 0.0)
    if value <= 0.0:
        return 0.0
    return value / (value + float(scale))


def _event_peak(event, default=None):
    for key in ("peak_frame", "end_frame", "start_frame"):
        v = event.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return default


def _ball_focus_peak(ball_detections):
    """Pic d'action réel = médiane des frames où un ballon est détecté.

    Renvoie ``None`` si aucune détection réelle de ballon : aucun pic inventé.
    """
    frames = sorted(int(b["frame"]) for b in (ball_detections or [])
                    if isinstance(b, dict) and b.get("frame") is not None)
    if not frames:
        return None
    n = len(frames)
    return frames[n // 2] if n % 2 else (frames[n // 2 - 1] + frames[n // 2]) // 2


def _event_focus_peak(events):
    """Fallback déterministe : médiane des pics d'événements (preuve de mouvement)."""
    peaks = sorted(p for p in (_event_peak(e) for e in (events or [])) if p is not None)
    if not peaks:
        return None
    n = len(peaks)
    return peaks[n // 2] if n % 2 else (peaks[n // 2 - 1] + peaks[n // 2]) // 2


def _track_centers(frames):
    """``{frame: (cx, cy)}`` à partir d'une trajectoire réelle (centres de bbox)."""
    centers = {}
    for i, f in enumerate(frames or []):
        b = _bbox_of(f)
        if b is None:
            continue
        centers[_frame_of(f, i)] = (
            (float(b[0]) + float(b[2])) / 2.0,
            (float(b[1]) + float(b[3])) / 2.0,
        )
    return centers


def _ball_proximity(centers, ball_detections, scale):
    """Distance minimale réelle sujet↔ballon (px) → score dans ``[0, 1]``.

    Le ballon doit **exister réellement** (``sports ball`` COCO). Pour chaque
    détection, on compare la position du ballon à la position du sujet à la même
    frame, ou à la frame connue la plus proche (maintien honnête). Renvoie
    ``(score, min_dist_px)`` ; ``(0.0, None)`` si aucune preuve exploitable.
    """
    if not centers or not ball_detections:
        return 0.0, None
    cframes = sorted(centers)
    best = None
    for b in ball_detections:
        if not isinstance(b, dict) or b.get("bbox") is None or b.get("frame") is None:
            continue
        try:
            bf = int(b["frame"])
        except (TypeError, ValueError):
            continue
        bb = b["bbox"]
        bx = (float(bb[0]) + float(bb[2])) / 2.0
        by = (float(bb[1]) + float(bb[3])) / 2.0
        # frame de sujet connue la plus proche (maintien de la dernière connue).
        near = min(cframes, key=lambda fr: (abs(fr - bf), fr))
        cx, cy = centers[near]
        d = math.hypot(cx - bx, cy - by)
        if best is None or d < best:
            best = d
    if best is None:
        return 0.0, None
    return math.exp(-best / float(scale)), float(best)


def _spatial_relevance(stat, height):
    """Pertinence spatiale réelle = position verticale normalisée du sujet.

    Un sujet plus bas dans l'image est au **premier plan** (plus proche de la
    caméra), donc plus « héros ». Signal réel, borné dans ``[0, 1]``.
    """
    if not height or stat is None:
        return 0.0
    my = float(stat.get("mean_y", 0.0))
    if my <= 0.0:
        return 0.0
    return max(0.0, min(1.0, my / float(height)))


# ---------------------------------------------------------------------------
# Classement multi-preuves
# ---------------------------------------------------------------------------
def rank_candidates(events, track_stats=None, tracks=None, ball_detections=None,
                    width=None, height=None, focus_peak=None,
                    disp_scale=DISP_SCALE, pers_scale=PERS_SCALE,
                    prox_scale=PROX_SCALE, ball_scale=BALL_SCALE):
    """Renvoie la liste classée (descendante) des candidats, chacun **explicable**.

    Chaque candidat expose ``score`` et ``breakdown`` (score par signal, poids,
    contribution) ainsi que les valeurs brutes (persistance, amplitude, distance
    ballon…). Le classement est total et stable : aucune dépendance à l'ordre
    d'entrée.
    """
    events = list(events or [])
    track_stats = track_stats or {}
    tracks = tracks or {}

    has_ball = bool(ball_detections)
    weights = dict(BASE_WEIGHTS)
    if has_ball:
        total = sum(weights.values()) + BALL_WEIGHT
        weights = {k: v / total for k, v in weights.items()}
        weights["ball_proximity"] = BALL_WEIGHT / total
    # sinon : BASE_WEIGHTS somme déjà à 1.0 (documenté).

    if focus_peak is None:
        # Priorité à l'action réelle (ballon) ; sinon médiane des pics d'événements.
        focus_peak = _ball_focus_peak(ball_detections)
        if focus_peak is None:
            focus_peak = _event_focus_peak(events)

    ranked = []
    for e in events:
        tid = _track_id(e)
        stat = track_stats.get(tid) if tid is not None else None

        persistence_frames = int(stat["len"]) if stat else 0
        amplitude_px = _displacement(e)
        if amplitude_px <= 0.0 and stat:
            amplitude_px = float(stat.get("displacement", 0.0))
        continuity = float(stat.get("continuity", 0.0)) if stat else 0.0
        event_conf = max(0.0, min(1.0, float(e.get("confidence", 0.0))))
        spatial = _spatial_relevance(stat, height)

        peak = _event_peak(e)
        if focus_peak is None or peak is None:
            proximity = 0.0
        else:
            proximity = math.exp(-abs(peak - focus_peak) / float(prox_scale))

        signals = {
            "amplitude": _sat(amplitude_px, disp_scale),
            "persistence": _sat(persistence_frames, pers_scale),
            "continuity": continuity,
            "event_proximity": proximity,
            "spatial_relevance": spatial,
            "event_confidence": event_conf,
        }
        ball_score, ball_dist = 0.0, None
        if has_ball and tid is not None and tracks.get(tid):
            ball_score, ball_dist = _ball_proximity(
                _track_centers(tracks.get(tid)), ball_detections, ball_scale)
            signals["ball_proximity"] = ball_score

        score = sum(weights[k] * signals.get(k, 0.0) for k in weights)
        breakdown = {
            k: {
                "value": round(float(signals.get(k, 0.0)), 6),
                "weight": round(float(weights[k]), 6),
                "contribution": round(float(weights[k] * signals.get(k, 0.0)), 6),
            }
            for k in weights
        }
        ranked.append({
            "event": e,
            "track_id": tid,
            "score": score,
            "signals": {k: float(signals.get(k, 0.0)) for k in weights},
            "breakdown": breakdown,
            "raw": {
                "persistence_frames": persistence_frames,
                "amplitude_px": round(float(amplitude_px), 2),
                "continuity": round(float(continuity), 4),
                "event_confidence": round(event_conf, 4),
                "spatial_relevance": round(float(spatial), 4),
                "event_peak_frame": peak,
                "ball_proximity_score": round(float(ball_score), 4),
                "ball_min_dist_px": (round(ball_dist, 2) if ball_dist is not None else None),
            },
            "_sort": (round(score, 12), float(persistence_frames),
                      round(float(amplitude_px), 6), str(e.get("event_id", ""))),
        })

    # Clé totale : score, persistance, amplitude, puis event_id décroissant.
    # ``reverse=True`` => event_id maximal en cas d'égalité parfaite (déterministe).
    ranked.sort(key=lambda c: c["_sort"], reverse=True)
    return ranked, {
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "ball_evidence": has_ball,
        "focus_peak_frame": focus_peak,
    }


def score_hero(events, track_stats=None, tracks=None, ball_detections=None,
               width=None, height=None, focus_peak=None):
    """Sélectionne le moment héros + le ``track_id`` suivi (contrat caméra).

    Renvoie un dict explicable : ``event``, ``track_id`` (== ``hero_track``),
    ``score``, ``selection`` (breakdown par signal), ``weights``, ``focus_peak_frame``
    et la liste ``ranked`` (top-N) pour l'audit.
    """
    if not events:
        return {"status": "UNKNOWN", "reason": "NO_EVIDENCE"}

    ranked, meta = rank_candidates(
        events, track_stats=track_stats, tracks=tracks,
        ball_detections=ball_detections, width=width, height=height,
        focus_peak=focus_peak)
    best = ranked[0]
    best_event = best["event"]
    hero_track = best["track_id"]

    return {
        "status": "OK",
        "event": best_event,
        "score": round(float(best["score"]), 6),
        "track_id": hero_track,
        # Alias explicite du contrat : ``hero_moment.hero_track`` DOIT égaler
        # ``camera_reframe.followed_track``. ``track_id`` est conservé pour la
        # compatibilité ascendante (même valeur).
        "hero_track": hero_track,
        "method": "multi_evidence_v2",
        "ranked_by": ["score", "persistence", "amplitude", "event_id"],
        "candidates": len(events),
        "weights": meta["weights"],
        "ball_evidence": meta["ball_evidence"],
        "focus_peak_frame": meta["focus_peak_frame"],
        "selection": {
            "persistence_frames": best["raw"]["persistence_frames"],
            "amplitude_px": best["raw"]["amplitude_px"],
            "continuity": best["raw"]["continuity"],
            "event_confidence": best["raw"]["event_confidence"],
            "spatial_relevance": best["raw"]["spatial_relevance"],
            "ball_proximity_score": best["raw"]["ball_proximity_score"],
            "ball_min_dist_px": best["raw"]["ball_min_dist_px"],
            "event_peak_frame": best["raw"]["event_peak_frame"],
            "breakdown": best["breakdown"],
        },
        "ranked": [
            {
                "event_id": c["event"].get("event_id"),
                "track_id": c["track_id"],
                "score": round(float(c["score"]), 6),
                "signals": {k: round(v, 6) for k, v in c["signals"].items()},
                "raw": c["raw"],
            }
            for c in ranked[:10]
        ],
    }
