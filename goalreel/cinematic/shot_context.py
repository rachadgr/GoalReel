"""Cut-aware evidence model + shot segmentation — GoalReel Director V2.3.

Rôle
----
Ce module introduit la distinction **fondamentale** que le Director V2.2 ne
faisait pas :

  A) une **fragmentation de suivi temporaire** (le même joueur change de
     ``track_id`` : occlusion, croisement, raté du détecteur) — traitée par
     Subject Continuity V2.2 ;
  B) une **vraie frontière de plan** (un montage : la caméra change d'axe /
     d'échelle) — traitée ici.

Sur la source actuelle, la frame **332** est une coupe RÉELLE mesurée (pic de
différence inter-frame réel, jamais supposé). Le Director V2.3 doit consommer
cette mesure pour **planifier** : une vraie coupe est une frontière cinématique
légitime, pas un « handoff » de suivi à lisser.

Politique de vérité
-------------------
  * La confiance d'une coupe est calculée à partir de **termes mesurés**
    réellement disponibles (pic de mouvement, variation de scène, discontinuité
    de suivi, changement d'apparence OSNet). Aucune valeur n'est inventée : les
    termes indisponibles sont exclus et les poids **renormalisés**.
  * Une coupe n'est déclarée ``REAL_CUT`` que si la confiance mesurée atteint
    ``REAL_CUT_MIN_CONF``. Sinon elle reste ``SOFT_TRANSITION``.
  * La **relation d'identité** à travers une coupe est calculée séparément de la
    pertinence cinématique : un plan post-coupe peut être
    **cinématiquement pertinent sans être la même identité**. On n'écrit
    ``SAME_CANONICAL_IDENTITY`` que si la continuité le **prouve**.

Déterminisme
------------
Toutes les itérations de dictionnaire sont triées ; les choix sont tranchés par
des clés totales ; les flottants sont arrondis de façon reproductible.
"""
from __future__ import annotations

import math
from typing import Any

from .edit_plan import MIN_SHOT_FRAMES, round_scale
from .evidence_model import is_person_sized
from ..tracking.continuity import appearance_representation, identity_containing

# ---------------------------------------------------------------------------
# Vocabulaire cut-aware (public).
# ---------------------------------------------------------------------------
REAL_CUT = "REAL_CUT"
SOFT_TRANSITION = "SOFT_TRANSITION"

SAME_CANONICAL_IDENTITY = "SAME_CANONICAL_IDENTITY"
NEW_IDENTITY = "NEW_IDENTITY"
UNKNOWN = "UNKNOWN"

CUT_MODEL_SCHEMA = "goalreel.cut_model.v2.3"
BOUNDARY_CONTINUOUS = "CONTINUOUS"
BOUNDARY_REAL_CUT = "REAL_CUT"

# Saturations du modèle de coupe (documentées, non spécifiques à une vidéo).
MOTION_PEAK_RATIO_FULL = 4.0   # un pic = 4x la base locale => motion_change = 1.0
_MOTION_LOCAL_WINDOW = 6       # fenêtre locale de base (frames)
_EPS = 1e-9

# Poids du modèle de coupe (renormalisés sur les termes RÉELLEMENT disponibles).
CUT_WEIGHTS: dict[str, float] = {
    "motion_change": 0.45,
    "scene_change": 0.25,
    "appearance_change": 0.15,
    "tracking_discontinuity": 0.15,
}

# Confiance minimale pour qu'une coupe mesurée soit une frontière RÉELLE.
REAL_CUT_MIN_CONF = 0.60

# Types de plans autorisés après une coupe réelle (preuve requise).
POST_CUT_TYPES: tuple[str, ...] = ("SECONDARY", "REACTION", "CELEBRATION",
                                   "FINAL_HERO", "CUTAWAY")


# ---------------------------------------------------------------------------
# Helpers déterministes
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


def _cosine(a: Any, b: Any) -> float | None:
    """Cosine de deux vecteurs réels L2-normalisés (``None`` si indisponible)."""
    if a is None or b is None:
        return None
    try:
        import numpy as np
        va = np.asarray(a, dtype="float64").ravel()
        vb = np.asarray(b, dtype="float64").ravel()
    except Exception:
        return None
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return None
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na <= _EPS or nb <= _EPS:
        return None
    return float(float(np.dot(va, vb)) / (na * nb))


def track_ranges(tracks: dict[Any, list] | None) -> dict[int, tuple[int, int]]:
    """``{track_id: (first_frame, last_frame)}`` (déterministe, réel)."""
    out: dict[int, tuple[int, int]] = {}
    for tid in sorted((tracks or {}), key=lambda k: int(k)):
        fs = [int(it["frame"]) for it in (tracks[tid] or [])
              if isinstance(it, dict) and it.get("frame") is not None]
        if fs:
            out[int(tid)] = (min(fs), max(fs))
    return out


def _centers_of(tracks: dict[Any, list], tid: int) -> dict[int, tuple[float, float]]:
    """``{frame: (cx, cy)}`` des centres RÉELS d'un fragment."""
    out: dict[int, tuple[float, float]] = {}
    for it in (tracks.get(tid) or []):
        if not isinstance(it, dict) or it.get("frame") is None or it.get("bbox") is None:
            continue
        b = it["bbox"]
        try:
            out[int(it["frame"])] = ((float(b[0]) + float(b[2])) / 2.0,
                                     (float(b[1]) + float(b[3])) / 2.0)
        except (TypeError, ValueError, IndexError):
            continue
    return dict(sorted(out.items()))


def moving_range(tracks: dict[Any, list] | None, track_id: int | None,
                 eps: float = 1e-3) -> tuple[int, int] | None:
    """Plage RÉELLE de mouvement d'un fragment (fin = dernier frame non gelé).

    ByteTrack maintient parfois la **dernière bbox connue** lorsqu'un sujet
    sort du champ : les frames « gelées » (centre identique à la précédente) ne
    sont PAS de la preuve de suivi et ne doivent pas servir à cadrer un plan.
    On renvoie ``(first, last_moving_frame)`` : dernier frame dont le centre
    diffère du précédent. ``None`` si le fragment n'existe pas.
    """
    if tracks is None or track_id is None or int(track_id) not in {int(k) for k in tracks}:
        return None
    c = _centers_of(tracks, int(track_id))
    keys = sorted(c)
    if not keys:
        return None
    last_moving = keys[0]
    for a, b in zip(keys, keys[1:]):
        ca, cb = c[a], c[b]
        if abs(cb[0] - ca[0]) > eps or abs(cb[1] - ca[1]) > eps:
            last_moving = b
    return int(keys[0]), int(last_moving)



def measure_cut(motion: list[float] | None, frame_index: int) -> dict[str, Any]:
    """Mesure le pic de mouvement à ``frame_index`` sur le profil réel.

    ``motion_change`` = ratio pic / base locale saturé ; ``scene_change`` =
    variation relative du contenu. Ces deux termes proviennent **exclusivement**
    du profil de mouvement mesuré sur la source.
    """
    motion = motion or []
    f = int(frame_index)
    if not motion or f <= 0 or f >= len(motion):
        return {"available": False, "motion_value": 0.0, "local_baseline": 0.0,
                "peak_ratio": 0.0, "motion_change": 0.0, "scene_change": 0.0}
    m = float(motion[f])
    lo = max(0, f - _MOTION_LOCAL_WINDOW)
    seg = [float(v) for v in motion[lo:f]]
    base = (sum(seg) / len(seg)) if seg else 0.0
    ratio = m / max(base, _EPS)
    return {
        "available": True,
        "motion_value": round_scale(m, 4),
        "local_baseline": round_scale(base, 4),
        "peak_ratio": round_scale(ratio, 4),
        "motion_change": round_scale(_clamp01(ratio / MOTION_PEAK_RATIO_FULL), 4),
        "scene_change": round_scale(_clamp01((m - base) / max(_EPS, m + base)), 4),
    }


def tracking_discontinuity(tracks: dict[Any, list] | None,
                           cut_frame: int) -> dict[str, Any]:
    """Discontinuité de suivi autour de ``cut_frame`` (preuve réelle).

    ``value`` = part des fragments qui **démarrent** à/après la coupe parmi ceux
    touchant la frontière (démarrent à/après + traversent). Un fragment qui
    traverse la coupe est une continuité de suivi ; un fragment qui apparaît
    après est une rupture. Aucune valeur inventée.
    """
    rng = track_ranges(tracks)
    c = int(cut_frame)
    starting_after = sorted(t for t, (f, _) in rng.items() if f >= c)
    spanning = sorted(t for t, (f, l) in rng.items() if f < c <= l)
    denom = len(starting_after) + len(spanning)
    value = (len(starting_after) / denom) if denom else 0.0
    return {
        "available": bool(rng),
        "value": round_scale(_clamp01(value), 4),
        "tracks_starting_after_cut": starting_after,
        "tracks_spanning_cut": spanning,
    }


def appearance_change(embeddings: dict[int, Any] | None,
                      tracks: dict[Any, list] | None,
                      cut_frame: int) -> dict[str, Any]:
    """Changement d'apparence OSNet à travers la coupe (preuve réelle).

    Compare la représentation d'apparence (moyenne L2-normalisée des embeddings
    réels) des fragments **avant** la coupe à celle des fragments **après**.
    ``value`` = ``(1 - cos) / 2`` ∈ [0,1] (0 = même apparence, 1 = opposée).
    Renvoie ``available=False`` si aucun embedding réel n'est disponible.
    """
    if not embeddings:
        return {"available": False, "value": None, "cosine": None,
                "pre_tracks": [], "post_tracks": []}
    rng = track_ranges(tracks)
    c = int(cut_frame)
    pre = sorted(t for t, (_, l) in rng.items() if l < c)
    post = sorted(t for t, (f, _) in rng.items() if f >= c)
    ref_pre = appearance_representation(embeddings, pre)
    ref_post = appearance_representation(embeddings, post)
    cos = _cosine(ref_pre, ref_post)
    if cos is None:
        return {"available": False, "value": None, "cosine": None,
                "pre_tracks": pre, "post_tracks": post}
    return {
        "available": True,
        "value": round_scale(_clamp01((1.0 - cos) / 2.0), 4),
        "cosine": round_scale(cos, 4),
        "pre_tracks": pre,
        "post_tracks": post,
    }


def build_cut_boundaries(scene: dict[str, Any] | None,
                         tracks: dict[Any, list] | None = None,
                         embeddings: dict[int, Any] | None = None
                         ) -> list[dict[str, Any]]:
    """Construit les frontières de plan mesurées (repr. propre des coupes).

    Pour chaque coupe réelle mesurée par :func:`scan_scene`, on assemble une
    représentation explicite ``{cut_frame, confidence, evidence, type}`` dont la
    confiance est calculée à partir des termes réellement disponibles.
    """
    scene = scene or {}
    motion = scene.get("motion") or []
    cuts = sorted({int(c) for c in (scene.get("cut_frames") or [])})
    out: list[dict[str, Any]] = []
    for c in cuts:
        mc = measure_cut(motion, c)
        td = tracking_discontinuity(tracks, c)
        ap = appearance_change(embeddings, tracks, c)
        terms = {
            "motion_change": mc["motion_change"] if mc["available"] else None,
            "scene_change": mc["scene_change"] if mc["available"] else None,
            "tracking_discontinuity": td["value"] if td["available"] else None,
            "appearance_change": ap["value"] if ap["available"] else None,
        }
        num = den = 0.0
        for key, w in CUT_WEIGHTS.items():
            v = terms.get(key)
            if v is None:
                continue
            num += w * float(v)
            den += w
        conf = (num / den) if den > 0 else 0.0
        out.append({
            "schema": CUT_MODEL_SCHEMA,
            "cut_frame": int(c),
            "confidence": round_scale(conf, 4),
            "type": REAL_CUT if conf >= REAL_CUT_MIN_CONF else SOFT_TRANSITION,
            "evidence": {
                "motion_change": terms["motion_change"],
                "appearance_change": terms["appearance_change"],
                "tracking_discontinuity": terms["tracking_discontinuity"],
                "scene_change": terms["scene_change"],
                "available": {k: (v is not None) for k, v in terms.items()},
                "motion_detail": mc,
                "tracking_detail": td,
                "appearance_detail": {"available": ap["available"],
                                      "cosine": ap["cosine"]},
            },
            "evidence_source": "VISIBLE_FROM_SOURCE",
        })
    return out


# ---------------------------------------------------------------------------
# Segmentation en contextes de plan
# ---------------------------------------------------------------------------
def _context_bounds(cut_frames: list[int], total_frames: int) -> list[tuple[int, int]]:
    cuts = sorted({int(c) for c in cut_frames if 0 < int(c) < int(total_frames)})
    starts = [0] + cuts
    out: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else (int(total_frames) - 1)
        if e >= s:
            out.append((s, e))
    return out


def _primary_subject(tracks: dict[Any, list], start: int, end: int,
                     hero_track: int | None,
                     moving: dict[int, tuple[int, int] | None] | None = None
                     ) -> int | None:
    """Sujet principal réel d'un contexte (héros s'il y est, sinon dominant).

    Le héros n'est retenu que si son **mouvement réel** recouvre le contexte
    (un tail gelé ByteTrack n'est pas de la preuve de suivi).
    """
    rng = track_ranges(tracks)
    rng_mv = moving if moving is not None else {tid: moving_range(tracks, tid)
                                                for tid in rng}
    if hero_track is not None:
        r = rng_mv.get(int(hero_track))
        if r and r[0] <= end and r[1] >= start:
            return int(hero_track)
    best: int | None = None
    best_len = -1
    for tid, r in rng_mv.items():
        if not r:
            continue
        f, l = r
        if l < start or f > end:
            continue
        if not is_person_sized(tracks[tid]):
            continue
        length = min(l, end) - max(f, start) + 1
        if length > best_len or (length == best_len and (best is None or tid < best)):
            best, best_len = int(tid), length
    return best


def identity_key(track_id: int | None, hero_identity: dict[str, Any] | None,
                 identity_by_track: dict[int, int]) -> int | None:
    """Clé d'identité canonique d'un fragment (``None`` si NON résolue).

    Un fragment isolé qui n'appartient à aucune identité canonique **résolue**
    (ni à la lignée héros) n'a **pas** de clé : sa relation à travers une coupe
    reste ``UNKNOWN`` — on ne revendique aucune identité.
    """
    if track_id is None:
        return None
    hero_canonical = (hero_identity or {}).get("canonical_track_id")
    hero_ids = {int(t) for t in (hero_identity or {}).get("source_track_ids", [])}
    if hero_canonical is not None and int(track_id) in hero_ids:
        return int(hero_canonical)
    return identity_by_track.get(int(track_id))


def identity_by_track(tracks: dict[Any, list] | None,
                      continuity: dict[str, Any] | None) -> dict[int, int]:
    """``{raw_track_id: canonical_identity_id}`` des identités **résolues**.

    Seuls les fragments réellement regroupés dans une identité canonique
    (``continuity['identities']``) sont présents. Les fragments non regroupés
    sont volontairement absents (leur relation reste ``UNKNOWN``).
    """
    out: dict[int, int] = {}
    for ident in (continuity or {}).get("identities", []) or []:
        canon = ident.get("canonical_track_id")
        for tid in ident.get("source_track_ids", []) or []:
            if canon is not None:
                out[int(tid)] = int(canon)
    return out


def relate(key_prev: int | None, key_cur: int | None) -> str:
    """Relation d'identité entre deux contextes (preuve requise).

    Cas particuliers honnêtes :

      * mêmes identités résolues => ``SAME_CANONICAL_IDENTITY`` ;
      * identités résolues différentes => ``NEW_IDENTITY`` ;
      * une identité seulement résolue (l'autre est un fragment non résolu,
        typiquement un contexte post-coupe dont la continuité n'est pas établie)
        => ``UNKNOWN`` — on ne **revendique** ni la même identité ni une
        identité nouvelle sans preuve ;
      * aucune identité résolue => ``UNKNOWN``.
    """
    if key_prev is None or key_cur is None:
        return UNKNOWN
    return SAME_CANONICAL_IDENTITY if int(key_prev) == int(key_cur) else NEW_IDENTITY



def shot_contexts(scene: dict[str, Any] | None,
                  tracks: dict[Any, list] | None,
                  *, hero_track: int | None = None,
                  hero_event: dict[str, Any] | None = None,
                  hero_identity: dict[str, Any] | None = None,
                  continuity: dict[str, Any] | None = None,
                  boundaries: list[dict[str, Any]] | None = None,
                  total_frames: int = 0,
                  fps: float = 30.0) -> list[dict[str, Any]]:
    """Partitionne la source en **contextes de plan** autour des coupes réelles.

    Chaque contexte porte : ``start_frame``, ``end_frame``, ``duration``
    (frames + secondes), ``active_tracks``, ``hero_evidence``,
    ``event_evidence``, ``cut_before``, ``cut_after``, ``confidence`` et la
    ``identity_relation`` avec le contexte précédent.
    """
    scene = scene or {}
    tracks = tracks or {}
    if total_frames <= 0:
        total_frames = int(scene.get("frames") or 0)
    if total_frames <= 0:
        return []
    if boundaries is None:
        boundaries = build_cut_boundaries(scene, tracks)
    real_cuts = sorted(int(b["cut_frame"]) for b in boundaries
                       if b.get("type") == REAL_CUT)
    conf_by_cut = {int(b["cut_frame"]): float(b["confidence"]) for b in boundaries}
    bounds = _context_bounds(real_cuts, total_frames)
    id_by_track = identity_by_track(tracks, continuity)
    rng = track_ranges(tracks)
    # Plage de MOUVEMENT réelle (le tail gelé ByteTrack n'est pas de la preuve).
    moving = {tid: moving_range(tracks, tid) for tid in rng}

    contexts: list[dict[str, Any]] = []
    prev_key: int | None = None
    for i, (s, e) in enumerate(bounds):
        active = sorted(t for t, (f, l) in rng.items() if not (l < s or f > e))
        hero_mv = moving.get(int(hero_track)) if hero_track is not None else None
        hero_range = rng.get(int(hero_track)) if hero_track is not None else None
        hero_here = bool(hero_mv and hero_mv[0] <= e and hero_mv[1] >= s)
        subject = _primary_subject(tracks, s, e, hero_track, moving=moving)
        key = identity_key(subject, hero_identity, id_by_track)
        cut_before = int(s) if i > 0 else None
        cut_after = int(bounds[i + 1][0]) if i + 1 < len(bounds) else None
        conf = 1.0
        if cut_before is not None:
            conf = conf_by_cut.get(cut_before, 0.0)
        contexts.append({
            "schema": CUT_MODEL_SCHEMA,
            "shot_context_id": f"SHOT_CONTEXT_{i + 1:02d}",
            "index": int(i),
            "start_frame": int(s),
            "end_frame": int(e),
            "duration_frames": int(e - s + 1),
            "duration_s": round_scale((e - s + 1) / float(fps or 30.0), 4),
            "active_tracks": active,
            "hero_evidence": {
                "hero_track": (int(hero_track) if hero_track is not None else None),
                "present": hero_here,
                "moving_range": ([int(hero_mv[0]), int(hero_mv[1])] if hero_mv else None),
                "range": ([int(hero_range[0]), int(hero_range[1])] if hero_range else None),
                "frozen_tail": (bool(hero_range and hero_mv
                                     and hero_range[1] > hero_mv[1])),
            },
            "event_evidence": (
                {"event_id": (hero_event or {}).get("event_id"),
                 "kind": (hero_event or {}).get("kind"),
                 "present": bool(hero_event)} if hero_event else {"present": False}),
            "cut_before": cut_before,
            "cut_after": cut_after,
            "confidence": round_scale(conf, 4),
            "boundary_before": (BOUNDARY_REAL_CUT if i > 0 else BOUNDARY_CONTINUOUS),
            "primary_subject": subject,
            "identity_key": key,
            "identity_relation": UNKNOWN if i == 0 else relate(prev_key, key),
        })
        prev_key = key
    return contexts


def context_of_frame(contexts: list[dict[str, Any]], frame: int) -> str | None:
    """``shot_context_id`` contenant ``frame`` (ou ``None``)."""
    for ctx in contexts:
        if int(ctx["start_frame"]) <= int(frame) <= int(ctx["end_frame"]):
            return str(ctx["shot_context_id"])
    return None


# ---------------------------------------------------------------------------
# Découpage des segments du Director aux coupes réelles
# ---------------------------------------------------------------------------
def split_segments_at_cuts(segments: list[tuple[int, int, str]],
                           cut_frames: list[int],
                           min_frames: int = MIN_SHOT_FRAMES,
                           ) -> tuple[list[tuple[int, int, str]], list[dict[str, Any]]]:
    """Découpe les segments du Director à chaque coupe RÉELLE interne.

    Une coupe est une frontière physique : un segment du director qui la
    chevauche est scindé en deux, de part et d'autre de la coupe. Si le
    découpage produirait un plan plus court que ``min_frames`` (coupe trop
    proche d'un bord), le segment est **laissé intact** : la coupe reste
    enregistrée comme frontière (le renderer interdit alors tout blend à
    travers elle) mais aucune frontière sous-granulaire n'est forcée.

    Renvoie ``(nouveaux_segments, journal)`` — partition préservée.
    """
    cuts = sorted({int(c) for c in cut_frames})
    out: list[tuple[int, int, str]] = []
    journal: list[dict[str, Any]] = []
    for (s, e, ph) in segments:
        inner = [c for c in cuts if int(s) < c <= int(e)]
        if not inner:
            out.append((int(s), int(e), ph))
            continue
        parts: list[tuple[int, int, str]] = []
        prev = int(s)
        for c in inner:
            parts.append((prev, c - 1, ph))
            prev = c
        parts.append((prev, int(e), ph))
        if all((b - a + 1) >= min_frames for a, b, _ in parts):
            out.extend(parts)
            journal.append({"segment": [int(s), int(e), ph], "cuts_applied": inner,
                            "parts": [[a, b] for a, b, _ in parts]})
        else:
            out.append((int(s), int(e), ph))
            journal.append({"segment": [int(s), int(e), ph], "cuts_applied": [],
                            "reason": "CUT_TOO_CLOSE_TO_SHOT_EDGE",
                            "cuts_near_edge": inner})
    return out, journal


def post_cut_track_pool(tracks: dict[Any, list] | None,
                        cut_frame: int) -> dict[int, list]:
    """Trajectoires **post-coupe** (fragment réel démarrant à/après la coupe).

    Ce sont les seules preuves autorisées pour cadrer un plan post-coupe : un
    plan après une vraie coupe ne peut PAS hériter du sujet d'avant-coupe.
    """
    rng = track_ranges(tracks)
    c = int(cut_frame)
    out: dict[int, list] = {}
    for tid in sorted(rng, key=lambda k: int(k)):
        if rng[tid][0] >= c:
            out[int(tid)] = (tracks or {})[tid]
    return out


def post_cut_candidate(tracks: dict[Any, list] | None, cut_frame: int,
                       lo: int, hi: int) -> int | None:
    """Meilleur candidat post-coupe **taille-personne** recouvrant ``[lo, hi]``.

    Déterministe : recouvrement décroissant, puis persistance, puis ``track_id``
    croissant. ``None`` si aucun candidat crédible (=> cadrage large honnête).
    """
    rng = track_ranges(tracks)
    c = int(cut_frame)
    best: int | None = None
    best_key: tuple[int, int, int] | None = None
    for tid in sorted(rng, key=lambda k: int(k)):
        f, l = rng[tid]
        if f < c:
            continue
        ov = min(l, int(hi)) - max(f, int(lo)) + 1
        if ov <= 0:
            continue
        if not is_person_sized((tracks or {})[tid]):
            continue
        key = (-ov, -(l - f + 1), int(tid))
        if best_key is None or key < best_key:
            best, best_key = int(tid), key
    return best


def identity_known(identity_by_track_map: dict[int, int], track_id: int | None) -> bool:
    """Vrai si le fragment appartient à une identité canonique résolue."""
    return track_id is not None and int(track_id) in identity_by_track_map


# ===========================================================================
# CUT-AWARE V2.3 — Porte de crédibilité post-coupe (déterministe, preuve)
# ===========================================================================
# Objectif : après une coupe RÉELLE, un sujet peut être visuellement DOMINANT
# (ex. une grande silhouette au premier plan) sans pour autant constituer une
# preuve cinématique de HERO / REACTION / CELEBRATION / FINAL_HERO. Cette porte
# sépare explicitement :
#
#   * SUBJECT DOMINANCE  — la plus grande bbox / la plus longue trajectoire ;
#   * CINEMATIC HERO EVIDENCE — preuve réellement compatible avec une phase.
#
# Aucune sémantique n'est inventée. Si la preuve est insuffisante, la phase est
# OMISE (au profit de CUTAWAY / SECONDARY), jamais forcée en HERO.
#
# Les termes de preuve utilisés sont tous MESURÉS (ou explicitement indisponibles) :
#   * taille « personne » du sujet (géométrie bbox réelle) ;
#   * mouvement réel non gelé (fin = dernier frame non gelé) ;
#   * persistance réelle (nombre de frames suivies) ;
#   * proximité au ballon RÉELLEMENT détecté (COCO 'sports ball') ;
#   * qualité de suivi (continuité / confiance) ;
#   * durée du contexte de plan ;
#   * relation d'identité canonique à travers la coupe (prouvée séparément).
# ---------------------------------------------------------------------------
CUTAWAY = "CUTAWAY"
PHASE_OMITTED = "OMITTED"

# Vocabulaire de relation d'identité (repris de edit_plan pour autonomie locale).
_REL_SAME = "SAME_CANONICAL_IDENTITY"
_REL_NEW = "NEW_IDENTITY"
_REL_UNKNOWN = "UNKNOWN"

# Seuils (documentés, non spécifiques à une vidéo).
REACTION_MIN_MOVING_FRAMES = 6      # un vrai mouvement de réaction (frames)
REACTION_MAX_DIST_PX = 260.0        # distance ballon réelle max (réaction au jeu)
CELEBRATION_MIN_TRACK_FRAMES = 18   # persistance minimale d'une célébration
FINAL_HERO_MIN_MOVING_FRAMES = 20   # mouvement réel minimal pour un final héros
FINAL_HERO_MIN_PERSISTENCE = 30     # persistance minimale pour un final héros
CONTEXT_TOO_SHORT_FRAMES = 12       # sous ce seuil, aucune phase n'est soutenue


def _saturate(value: float, scale: float) -> float:
    """Saturation douce monotone dans ``[0, 1)`` (``x / (x + scale)``)."""
    v = float(value or 0.0)
    if v <= 0.0:
        return 0.0
    return v / (v + float(scale))


def subject_ball_distance(tracks: dict[Any, list] | None, track_id: int | None,
                          ball_detections: list[dict] | None,
                          start: int, end: int) -> float | None:
    """Distance minimale RÉELLE entre un sujet suivi et un ballon détecté.

    On compare, pour chaque frame où les deux existent, le centre de bbox du
    sujet au centre du ballon détecté ; on renvoie la distance minimale. ``None``
    si aucune comparaison n'est possible (preuve indisponible).
    """
    if tracks is None or track_id is None or not ball_detections:
        return None
    import math
    centers = _centers_of(tracks, int(track_id))
    if not centers:
        return None
    best: float | None = None
    for b in ball_detections:
        if not isinstance(b, dict) or b.get("bbox") is None or b.get("frame") is None:
            continue
        try:
            f = int(b["frame"])
        except (TypeError, ValueError):
            continue
        if not (int(start) <= f <= int(end)) or f not in centers:
            continue
        bb = b["bbox"]
        try:
            bx = (float(bb[0]) + float(bb[2])) / 2.0
            by = (float(bb[1]) + float(bb[3])) / 2.0
        except (TypeError, ValueError, IndexError):
            continue
        cx, cy = centers[f]
        d = math.hypot(cx - bx, cy - by)
        if best is None or d < best:
            best = float(d)
    return best


def _subject_evidence(tracks: dict[Any, list], tid: int | None,
                      start: int, end: int) -> dict[str, Any]:
    """Preuves mesurées d'un fragment candidat sur un contexte ``[start, end]``."""
    if tid is None:
        return {
            "track_id": None, "present": False, "person_sized": False,
            "moving_frames": 0, "track_frames": 0, "continuity": 0.0,
            "moving_range": None, "range": None, "overlap": 0,
        }
    items = tracks.get(int(tid)) or []
    if not items:
        return {
            "track_id": int(tid), "present": False, "person_sized": False,
            "moving_frames": 0, "track_frames": 0, "continuity": 0.0,
            "moving_range": None, "range": None, "overlap": 0,
        }
    rng = track_ranges({tid: items}).get(int(tid))
    mv = moving_range(tracks, int(tid))
    ov = 0
    if rng:
        ov = max(0, min(rng[1], int(end)) - max(rng[0], int(start)) + 1)
    moving_frames = 0
    if mv:
        moving_frames = max(0, min(mv[1], int(end)) - max(mv[0], int(start)) + 1)
    track_frames = int(rng[1] - rng[0] + 1) if rng else 0
    cont = (len(items) / float(track_frames or 1)) if track_frames else 0.0
    return {
        "track_id": int(tid),
        "present": bool(ov > 0),
        "person_sized": bool(is_person_sized(items)),
        "moving_frames": int(moving_frames),
        "track_frames": int(track_frames),
        "continuity": round_scale(min(1.0, cont), 4),
        "moving_range": ([int(mv[0]), int(mv[1])] if mv else None),
        "range": ([int(rng[0]), int(rng[1])] if rng else None),
        "overlap": int(ov),
    }


def post_cut_phase_gate(tracks: dict[Any, list] | None,
                        cut_frame: int,
                        *,
                        subject: int | None = None,
                        start: int = 0, end: int = 0,
                        ball_detections: list[dict] | None = None,
                        identity_relation: str = _REL_UNKNOWN,
                        has_lineage: bool = False,
                        motion: list[float] | None = None,
                        fps: float = 30.0,
                        other_evidence: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
    """Porte déterministe de crédibilité pour une phase post-coupe.

    Renvoie ``{'phase', 'evidence', 'terms', 'omitted', 'reason'}`` :

      * ``HERO`` / ``FINAL_HERO`` : exige une preuve héros forte — sujet
        **taille-personne**, **mouvement réel** substantiel, **persistance**
        réelle et, si des ballons sont détectés, une proximité ballon
        raisonnable. La seule dominance visuelle (grande bbox) **ne suffit
        pas** : un blob large n'est jamais promu HERO ;
      * ``REACTION`` : exige un vrai mouvement **et** une proximité au jeu
        (ballon réellement détecté) — sinon ``None`` ;
      * ``CELEBRATION`` : exige une persistance réelle et un mouvement réel —
        sinon ``None`` ;
      * ``SECONDARY`` : preuve plus faible mais sujet taille-personne présent ;
      * ``CUTAWAY`` : aucun sujet taille-personne crédible (cadrage large honnête) ;
      * ``OMITTED`` : contexte trop court pour soutenir une phase.
    """
    tracks = tracks or {}
    start = int(start) if end else int(cut_frame)
    end = int(end) if end else int(cut_frame)
    ctx_len = max(0, end - start + 1)
    ev = _subject_evidence(tracks, subject, start, end)
    terms: dict[str, Any] = {
        "subject_present": bool(ev["present"]),
        "subject_person_sized": bool(ev["person_sized"]),
        "moving_frames": int(ev["moving_frames"]),
        "track_frames": int(ev["track_frames"]),
        "continuity": float(ev["continuity"]),
        "context_frames": int(ctx_len),
        "identity_relation": identity_relation,
        "has_lineage": bool(has_lineage),
    }

    # Preuve ballon réelle (indisponible => None, jamais inventée).
    ball_available = bool(ball_detections)
    ball_dist = subject_ball_distance(tracks, subject, ball_detections, start, end)
    terms["ball_evidence_available"] = bool(ball_available)
    terms["ball_min_distance_px"] = (round_scale(ball_dist, 3)
                                     if ball_dist is not None else None)

    # Contrainte honnête sur la relation d'identité : on ne revendique jamais
    # SAME_CANONICAL_IDENTITY sans lignée prouvée.
    same_identity = bool(identity_relation == _REL_SAME and has_lineage)

    def _result(phase: str | None, reason: str,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
        out = {
            "phase": phase,
            "omitted": phase is None,
            "reason": reason,
            "evidence": [f"gate:{reason}"],
            "terms": dict(terms),
        }
        if extra:
            out["terms"].update(extra)
        return out

    if ctx_len < CONTEXT_TOO_SHORT_FRAMES:
        return _result(None, "CONTEXT_TOO_SHORT")

    # --- Sujet absent ou non « taille-personne » ----------------------------
    if not ev["present"]:
        return _result(CUTAWAY, "NO_PERSON_SIZED_SUBJECT")
    if not ev["person_sized"]:
        # Dominance visuelle d'un blob large : JAMAIS promu HERO.
        return _result(CUTAWAY, "DOMINANT_BLOB_NOT_PERSON_SIZED")

    moving = int(ev["moving_frames"])
    pers = int(ev["track_frames"])

    # --- HERO / FINAL_HERO : preuve héros FORTE -----------------------------
    hero_evidence = (
        moving >= FINAL_HERO_MIN_MOVING_FRAMES
        and pers >= FINAL_HERO_MIN_PERSISTENCE
        and (not ball_available or ball_dist is None or ball_dist <= REACTION_MAX_DIST_PX)
    )
    if hero_evidence:
        terms["hero_evidence_strong"] = True
        # ``FINAL_HERO`` (retour au héros canonique) exige une continuité
        # d'identité PROUVÉE à travers la coupe ; sinon c'est un contexte héros
        # distinct (``HERO``) — jamais une continuité revendiquée sans preuve.
        phase = "FINAL_HERO" if same_identity else "HERO"
        return _result(phase, "STRONG_HERO_EVIDENCE", {"hero_evidence_strong": True})
    terms["hero_evidence_strong"] = False

    # --- REACTION : mouvement réel + proximité au jeu -----------------------
    reaction_ok = (
        moving >= REACTION_MIN_MOVING_FRAMES
        and (not ball_available or (ball_dist is not None and ball_dist <= REACTION_MAX_DIST_PX))
    )
    if reaction_ok:
        return _result("REACTION", "REACTION_EVIDENCE")

    # --- CELEBRATION : persistance + mouvement réels ------------------------
    celebration_ok = (
        pers >= CELEBRATION_MIN_TRACK_FRAMES
        and moving >= REACTION_MIN_MOVING_FRAMES
    )
    if celebration_ok:
        return _result("CELEBRATION", "CELEBRATION_EVIDENCE")

    # --- SECONDARY : sujet taille-personne mais preuve faible ---------------
    return _result("SECONDARY", "WEAK_SUBJECT_EVIDENCE")


def post_cut_segments(tracks: dict[Any, list] | None,
                      contexts: list[dict[str, Any]],
                      *,
                      ball_detections: list[dict] | None = None,
                      motion: list[float] | None = None,
                      fps: float = 30.0) -> tuple[list[tuple[int, int, str]],
                                                   list[dict[str, Any]]]:
    """Segments post-coupe dérivés de la PORTE DE CRÉDIBILITÉ (jamais HERO forcé).

    Un contexte de plan dont ``index > 0`` (donc précédé d'une coupe réelle) est
    évalué par :func:`post_cut_phase_gate`. Le segment produit porte la phase
    retenue (``REACTION`` / ``CELEBRATION`` / ``SECONDARY`` / ``CUTAWAY`` /
    ``FINAL_HERO``) ou est **omis** si le contexte est trop court. Renvoie
    ``(segments, journal)`` — déterministe.
    """
    tracks = tracks or {}
    segs: list[tuple[int, int, str]] = []
    journal: list[dict[str, Any]] = []
    for ctx in contexts:
        if int(ctx.get("index", 0)) <= 0:
            continue
        s = int(ctx["start_frame"])
        e = int(ctx["end_frame"])
        subject = ctx.get("primary_subject")
        rel = str(ctx.get("identity_relation", _REL_UNKNOWN))
        gate = post_cut_phase_gate(
            tracks, int(ctx.get("cut_before") or s),
            subject=(int(subject) if subject is not None else None),
            start=s, end=e, ball_detections=ball_detections,
            identity_relation=rel, has_lineage=(rel == _REL_SAME),
            motion=motion, fps=fps)
        entry = {
            "shot_context_id": ctx.get("shot_context_id"),
            "start_frame": s, "end_frame": e,
            "subject": subject, "identity_relation": rel,
            "phase": gate["phase"], "reason": gate["reason"],
            "omitted": bool(gate["omitted"]),
            "terms": gate["terms"],
        }
        journal.append(entry)
        # Un contexte dont la preuve est insuffisante n'est JAMAIS promu HERO :
        # il est couvert par un plan de coupe honnête (CUTAWAY, cadrage large),
        # ce qui préserve la partition du temps sans inventer de phase.
        phase = gate["phase"]
        if phase in (None, PHASE_OMITTED):
            phase = CUTAWAY
            entry["filled_as"] = CUTAWAY
        segs.append((s, e, str(phase)))
    return segs, journal


def format_phase_gate_evidence(gate: dict[str, Any]) -> list[str]:
    """Lignes de preuve sérialisables pour un plan post-coupe (audit)."""
    out = [f"phase_gate:{gate.get('reason')}"]
    terms = gate.get("terms", {}) or {}
    out.append(f"gate_subject:{terms.get('subject_person_sized')}")
    out.append(f"gate_moving_frames:{terms.get('moving_frames')}")
    out.append(f"gate_persistence:{terms.get('track_frames')}")
    if terms.get("ball_evidence_available"):
        out.append(f"gate_ball_dist:{terms.get('ball_min_distance_px')}")
    return out


__all__ = [
    "REAL_CUT", "SOFT_TRANSITION", "SAME_CANONICAL_IDENTITY", "NEW_IDENTITY",
    "UNKNOWN", "CUT_MODEL_SCHEMA", "BOUNDARY_CONTINUOUS", "BOUNDARY_REAL_CUT",
    "CUT_WEIGHTS", "REAL_CUT_MIN_CONF", "MOTION_PEAK_RATIO_FULL",
    "measure_cut", "tracking_discontinuity", "appearance_change",
    "build_cut_boundaries", "shot_contexts", "context_of_frame",
    "split_segments_at_cuts", "post_cut_track_pool", "post_cut_candidate",
    "identity_by_track", "identity_key", "track_ranges", "moving_range", "relate",
    # --- CUT-AWARE V2.3 : porte de crédibilité post-coupe -------------------
    "CUTAWAY", "PHASE_OMITTED", "post_cut_phase_gate", "post_cut_segments",
    "subject_ball_distance", "format_phase_gate_evidence",
    "REACTION_MIN_MOVING_FRAMES", "REACTION_MAX_DIST_PX",
    "CELEBRATION_MIN_TRACK_FRAMES", "FINAL_HERO_MIN_MOVING_FRAMES",
    "FINAL_HERO_MIN_PERSISTENCE", "CONTEXT_TOO_SHORT_FRAMES",
]
