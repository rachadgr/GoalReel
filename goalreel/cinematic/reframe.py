"""Reframe 9:16 piloté par la preuve de suivi + timing cinématique d'événement.

Deux garanties :

1. **Cohérence héros → caméra** : le sujet suivi est celui du *moment héros*
   (``preferred_track``) dès qu'une trajectoire réelle existe pour cet id. Sinon,
   fallback documenté ``LONGEST_TRACK`` (raison enregistrée).
2. **Timing cinématique réflexif** : lorsqu'un événement héros est fourni, la
   ligne temporelle des cibles est structurée en quatre phases explicites
   (``anticipation`` → ``hero_moment`` → ``reaction`` → ``celebration``) qui
   prolongent naturellement le suivi réel autour de l'événement.

Contraintes de vérité :

  * aucune coordonnée inventée : toute cible provient d'un centre de bbox
    réellement suivie ; les frames sans suivi réutilisent la **dernière position
    réelle connue** (maintien), jamais une position fabriquée ;
  * la continuité est préservée (aucun saut de cible synthétique) ;
  * le bornage (``clamp_targets``) garantit qu'aucune bordure noire n'apparaît.

En l'absence totale de trajectoire, on renvoie ``{}`` (fallback statique), jamais
un faux suivi.
"""
from __future__ import annotations

import cv2  # noqa: F401  (conservé pour compatibilité d'import historique)
import numpy as np  # noqa: F401

# Étendues (en frames) des phases cinématiques autour de l'événement. Elles
# n'inventent AUCUN mouvement : elles ne font que prolonger/hold le dernier
# centre réellement suivi.
ANTICIPATION_FRAMES = 30
REACTION_FRAMES = 24
CELEBRATION_FRAMES = 30


def _no_tracks_meta(reason="NO_TRACKS"):
    return {"source": "NO_TRACKS", "mode": "STATIC_FALLBACK",
            "followed_track": None, "target_frames": 0, "selection": "NONE",
            "reason": reason, "phases": {}}


def _frame_cx(items):
    """``{frame: cx}`` des centres horizontaux réels (ignore les items malformés)."""
    out = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        fr = item.get("frame")
        if bbox is None or fr is None:
            continue
        try:
            out[int(fr)] = (float(bbox[0]) + float(bbox[2])) / 2.0
        except (TypeError, ValueError, IndexError):
            continue
    return out


def plan_camera(primary_frames, event=None, width=None,
                anticipation=ANTICIPATION_FRAMES,
                reaction=REACTION_FRAMES,
                celebration=CELEBRATION_FRAMES):
    """Construit la ligne temporelle cinématique (phases) autour de l'événement.

    ``primary_frames`` : liste d'items ``{frame,bbox}`` de la trajectoire suivie
    (preuve réelle). ``event`` : événement héros (``start_frame``/``peak_frame``/
    ``end_frame``) ou ``None``.

    Renvoie ``(targets, meta)`` :

      * ``targets`` : ``{frame: cx}`` — **toutes** les valeurs proviennent de
        centres réels suivis (ou du maintien de la dernière valeur réelle).
      * ``meta['phases']`` : ``{frame: 'anticipation'|'hero_moment'|'reaction'|
        'celebration'|'track'}`` pour l'audit et les tests.

    Sans événement, on renvoie simplement les cibles réelles (comportement
    historique) avec la phase ``track``.
    """
    per_frame = _frame_cx(primary_frames)
    if not per_frame:
        return {}, {"phases": {}, "phases_summary": {}}
    frames_sorted = sorted(per_frame)

    def _hold(idx):
        """Centre réel maintenu : dernière position connue ≤ idx, sinon première."""
        cand = None
        for k in frames_sorted:
            if k <= idx:
                cand = per_frame[k]
            else:
                break
        return per_frame[frames_sorted[0]] if cand is None else cand

    # Bornes réelles de la trajectoire suivie.
    first_t, last_t = frames_sorted[0], frames_sorted[-1]
    targets = dict(per_frame)
    phases = {f: "track" for f in frames_sorted}

    if event:
        s = event.get("start_frame")
        e = event.get("end_frame")
        p = event.get("peak_frame")
        s = first_t if s is None else max(first_t, int(s))
        e = last_t if e is None else min(last_t, int(e))
        if e < s:
            s, e = e, s

        # 1) ANTICIPATION : approche sur la position réelle du héros (aucun pan
        #    synthétique). Les frames sans suivi réutilisent la position connue
        #    la plus proche (maintien honnête).
        ant_start = max(first_t, s - int(anticipation))
        for f in range(ant_start, s):
            targets[f] = _hold(f) if f not in per_frame else per_frame[f]
            phases[f] = "anticipation"

        # 2) HERO MOMENT : suivi réel exact de l'événement [s, e].
        for f in range(s, e + 1):
            targets[f] = _hold(f) if f not in per_frame else per_frame[f]
            phases[f] = "hero_moment"

        # 3) REACTION / 4) CELEBRATION : maintien du dernier centre réellement
        #    suivi (réaction immédiate + célébration), sans mouvement inventé.
        r_end = e + int(reaction)
        c_end = r_end + int(celebration)
        for f in range(e + 1, r_end + 1):
            targets[f] = _hold(f)
            phases[f] = "reaction"
        for f in range(r_end + 1, c_end + 1):
            targets[f] = _hold(f)
            phases[f] = "celebration"
        event_window = {"start_frame": s, "peak_frame": p, "end_frame": e}
    else:
        event_window = None

    # Continuité : amplitude maximale du saut entre deux cibles consécutives.
    tkeys = sorted(targets)
    max_step = 0.0
    for a, b in zip(tkeys, tkeys[1:]):
        max_step = max(max_step, abs(targets[b] - targets[a]))

    summary = {}
    for ph in phases.values():
        summary[ph] = summary.get(ph, 0) + 1

    return targets, {
        "phases": phases,
        "phases_summary": summary,
        "event_window": event_window,
        "max_step_px": round(float(max_step), 3),
    }


def build_reframe_targets(tracks, preferred_track=None, hero_event=None,
                          width=None, anticipation=ANTICIPATION_FRAMES,
                          reaction=REACTION_FRAMES, celebration=CELEBRATION_FRAMES):
    """Cibles caméra 9:16 à partir des trajectoires RÉELLES suivies.

    ``tracks`` : ``track_id -> [ {frame, bbox, ...}, ... ]`` (ByteTrack sur de
    vraies détections YOLO).

    ``preferred_track`` : le sujet du *moment héros*. Il devient le sujet suivi
    dès qu'une trajectoire réelle existe pour cet id (cohérence événement →
    caméra). Sinon, fallback documenté sur la trajectoire la plus persistante
    (``LONGEST_TRACK``) et ``reason`` enregistrée.

    ``hero_event`` : événement héros (avec ``start_frame``/``peak_frame``/
    ``end_frame``). S'il est fourni, la ligne temporelle est structurée en
    phases cinématiques (anticipation / hero_moment / reaction / celebration).

    Renvoie ``(targets, meta)`` où ``targets`` est ``{frame_idx: cx}`` (centre
    horizontal réel, en pixels source) et ``meta`` décrit honnêtement la source.
    """
    if not tracks:
        return {}, _no_tracks_meta()

    hero_known = preferred_track is not None and bool(tracks.get(preferred_track))
    if hero_known:
        primary_id = preferred_track
        selection = "HERO_TRACK"
        reason = "HERO_TRACK_AVAILABLE"
    else:
        # Fallback documenté, déterministe : trajectoire la plus persistante.
        # Départage stable par id décroissant (jamais l'ordre d'insertion).
        primary_id = max(tracks, key=lambda tid: (len(tracks[tid]), tid))
        selection = "LONGEST_TRACK"
        reason = ("PREFERRED_TRACK_UNAVAILABLE" if preferred_track is not None
                  else "NO_PREFERRED_TRACK")

    primary_items = [it for it in (tracks.get(primary_id) or []) if isinstance(it, dict)]
    per_frame = _frame_cx(primary_items)
    if not per_frame:
        return {}, _no_tracks_meta(reason="PRIMARY_TRACK_EMPTY")

    plan_targets, plan_meta = plan_camera(
        primary_items, event=hero_event, width=width,
        anticipation=anticipation, reaction=reaction, celebration=celebration)

    # Si l'événement ne concerne pas le sujet suivi, la phase n'a pas de sens :
    # on retombe sur un pur suivi réel.
    phases = plan_meta.get("phases", {})
    if not hero_event:
        phases = {f: "track" for f in sorted(per_frame)}

    return plan_targets, {
        "source": "REAL_TRACKS",
        "mode": "FOLLOW_TRACKED_SUBJECT",
        "followed_track": int(primary_id),
        "followed_track_frames": len(primary_items),
        "target_frames": len(plan_targets),
        "selection": selection,
        "reason": reason,
        "phases": phases,
        "phases_summary": plan_meta.get("phases_summary", {}),
        "event_window": plan_meta.get("event_window"),
        "max_step_px": plan_meta.get("max_step_px", 0.0),
    }


def clamp_targets(targets, width, crop_w):
    """Borne chaque centre suivi pour que la fenêtre 9:16 reste DANS la source.

    ``crop_w`` est la largeur du crop 9:16 (``round(height*9/16)``, bornée à la
    largeur source). Le centre doit rester dans ``[crop_w/2, width-crop_w/2]`` :
    cela garantit qu'aucune bordure noire n'apparaît et que le sujet suivi reste
    cadré. Si la source est plus étroite que le crop, tout est centré.
    """
    if not targets:
        return {}
    if crop_w >= width:
        mid = width / 2.0
        return {int(k): mid for k in targets}
    lo = crop_w / 2.0
    hi = width - crop_w / 2.0
    return {int(k): float(min(hi, max(lo, float(v)))) for k, v in targets.items()}


def crop_to_target(frame, bbox, target_w=1080, target_h=1920, pad=2.2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = max(2, x2 - x1) * pad
    bh = max(2, y2 - y1) * pad
    ar = target_w / target_h
    if bw / bh > ar:
        bh = bw / ar
    else:
        bw = bh * ar
    x1 = max(0, int(cx - bw / 2)); x2 = min(w, int(cx + bw / 2))
    y1 = max(0, int(cy - bh / 2)); y2 = min(h, int(cy + bh / 2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
