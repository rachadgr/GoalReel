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


__all__ = [
    "REAL_CUT", "SOFT_TRANSITION", "SAME_CANONICAL_IDENTITY", "NEW_IDENTITY",
    "UNKNOWN", "CUT_MODEL_SCHEMA", "BOUNDARY_CONTINUOUS", "BOUNDARY_REAL_CUT",
    "CUT_WEIGHTS", "REAL_CUT_MIN_CONF", "MOTION_PEAK_RATIO_FULL",
    "measure_cut", "tracking_discontinuity", "appearance_change",
    "build_cut_boundaries", "shot_contexts", "context_of_frame",
    "split_segments_at_cuts", "post_cut_track_pool", "post_cut_candidate",
    "identity_by_track", "identity_key", "track_ranges", "moving_range", "relate",
]
