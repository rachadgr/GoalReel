"""Cinematic Director V2 — planificateur déterministe piloté par la preuve.

Rôle
----
Transformer la preuve du pipeline existant (suivi réel, événements, héros
multi-preuves, ballon COCO, mouvement/coupes réels de la source) en un **plan de
montage explicite** consommé par le renderer (:mod:`goalreel.cinematic.render_plan`).

Le director est une **couche de décision** : il n'infère rien lui-même et ne
remplace aucun modèle. Il lit des mesures déjà produites et les organise en
plans cinématiques. C'est volontairement un module séparé des adapters de
modèles (« préférer une couche director/planning propre »).

Structure cible (émise **uniquement** quand la preuve le permet)
---------------------------------------------------------------
``HOOK → BUILD_UP → ANTICIPATION → ACTION → HERO → REACTION → CELEBRATION →
SECONDARY → CLIMAX → FINAL_HERO → OUTRO``

Toutes les phases absentes sont listées dans ``phases_absent`` **avec leur
raison** : aucun plan n'est fabriqué pour « remplir » la structure. Une vignette
plus courte mais réellement soutenue par la preuve est préférée à une structure
complète mais fabriquée.

Preuves consommées
------------------
  * ``hero`` (``multi_evidence_v2``) : sujet héros + ``track_id`` ;
  * ``tracks`` : trajectoires réellement suivies (ByteTrack) ;
  * profil de vitesse **du héros** (|Δcentre| lissé) : pic d'action réel, donc
    les beats ``HERO`` / ``ACTION`` / ``FINAL_HERO`` ne sont pas arbitraires ;
  * ``scene`` (:func:`goalreel.cinematic.evidence_model.scan_scene`) : coupes
    réelles + profil de mouvement global (CLIMAX, frontières de plans) ;
  * ``ball_detections`` : preuve ballon réelle (jamais inventée) ;
  * géométrie réelle du groupe (étendue horizontale) pour le cadrage.

Contrats garantis
-----------------
  * ``hero_track == followed_track`` ;
  * toute phase centrée héros a ``selected_track == hero_track`` ;
  * aucune coordonnée caméra n'est inventée (centres réellement suivis, maintien
    de la dernière position réelle connue) ;
  * durée totale préservée (voir ``time_map``) ;
  * déterminisme : mêmes preuves ⇒ même plan.
"""
from __future__ import annotations

import math
from typing import Any

from .edit_plan import (
    ACTION,
    ACTION_LEN,
    ANTICIPATION,
    ANTICIPATION_LEN,
    BUILD_UP,
    CAM_ANTICIPATION_FOLLOW,
    CAM_CELEBRATION_FRAMING,
    CAM_FINAL_HERO_FRAMING,
    CAM_PULL_BACK,
    CAM_PUSH_IN,
    CAM_REACTION_FRAMING,
    CAM_STABLE_FOLLOW,
    CAMERA_CROP_DELTA,
    CELEBRATION,
    CLIMAX,
    CLIMAX_HALF,
    CROP_FRACTION,
    CROP_FRACTION_MAX,
    CROP_FRACTION_MIN,
    EditPlan,
    FINAL_HERO,
    HERO,
    HERO_POST,
    HERO_PRE,
    HOOK,
    HOOK_FRAMES,
    MIN_SEGMENT_FRAMES,
    MIN_SHOT_FRAMES,
    OUTRO,
    OUTRO_FRAMES,
    PHASE_ORDER,
    REACTION,
    REACTION_MIN,
    SECONDARY,
    SHOT_CELEBRATION,
    SHOT_FINAL_HERO,
    SHOT_HERO_MEDIUM,
    SHOT_HOOK,
    SHOT_LOW_FEELING,
    SHOT_MEDIUM,
    SHOT_OUTRO,
    SHOT_REACTION,
    SHOT_WIDE,
    SPEED_MAX,
    SPEED_MIN,
    SUBJECT_Y_BIAS,
    Shot,
    round_scale,
)
from .evidence_model import (
    build_evidence,
    hero_presence,
    is_person_sized,
    track_frame_centers,
)
from .speed_design import speed_curve
from .transitions import resolve_transitions
from ..tracking.continuity import (
    active_track_at,
    analyze_continuity,
    hero_identity,
    identity_frames,
)

# ---------------------------------------------------------------------------
# Paramètres de montage (documentés, déterministes).
# ---------------------------------------------------------------------------
REACTION_LEN = 30              # ~1.0 s de réaction après le beat héros
FINAL_HERO_PRE = 6             # ~0.2 s avant le second beat réel
FINAL_HERO_POST = 17           # ~0.6 s après le second beat réel
SECOND_BURST_MIN_SEP = 24      # séparation minimale entre deux pics du héros
SECOND_BURST_MIN_RATIO = 0.30  # un 2e pic doit valoir >= 30 % du pic principal
SPEED_SMOOTH_WINDOW = 5        # lissage du profil de vitesse du héros (frames)

# Priorité de résolution des recouvrements (plus grand gagne). Totale et
# déterministe : aucune phase ne partage la même valeur.
PHASE_PRIORITY: dict[str, int] = {
    HERO: 100,
    OUTRO: 95,
    CLIMAX: 90,
    FINAL_HERO: 85,
    REACTION: 70,
    ACTION: 60,
    ANTICIPATION: 55,
    CELEBRATION: 50,
    SECONDARY: 40,
    BUILD_UP: 30,
    HOOK: 20,
}

# Cadrage / caméra par phase (documenté).
PHASE_FRAMING: dict[str, dict[str, str]] = {
    HOOK: {"shot": SHOT_HOOK, "camera": CAM_PULL_BACK},
    BUILD_UP: {"shot": SHOT_WIDE, "camera": CAM_STABLE_FOLLOW},
    ANTICIPATION: {"shot": SHOT_MEDIUM, "camera": CAM_ANTICIPATION_FOLLOW},
    ACTION: {"shot": SHOT_MEDIUM, "camera": CAM_STABLE_FOLLOW},
    HERO: {"shot": SHOT_HERO_MEDIUM, "camera": CAM_PUSH_IN},
    REACTION: {"shot": SHOT_REACTION, "camera": CAM_REACTION_FRAMING},
    CELEBRATION: {"shot": SHOT_CELEBRATION, "camera": CAM_CELEBRATION_FRAMING},
    SECONDARY: {"shot": SHOT_WIDE, "camera": CAM_STABLE_FOLLOW},
    CLIMAX: {"shot": SHOT_LOW_FEELING, "camera": CAM_CELEBRATION_FRAMING},
    FINAL_HERO: {"shot": SHOT_FINAL_HERO, "camera": CAM_FINAL_HERO_FRAMING},
    OUTRO: {"shot": SHOT_OUTRO, "camera": CAM_PULL_BACK},
}

# Phases centrées sur le héros : elles DOIVENT suivre ``hero_track``.
HERO_CENTRIC_PHASES = (ANTICIPATION, ACTION, HERO, REACTION, FINAL_HERO)

# Phases dont la vitesse est justifiée par la présence d'un événement réel.
EVENT_SPEED_PHASES = (ANTICIPATION, ACTION, HERO, REACTION, CELEBRATION, CLIMAX,
                      FINAL_HERO, BUILD_UP)

# Phase attendue pour chaque raison de fallback (audit ``phases_absent``).
_FALLBACK_PHASE: dict[str, str | None] = {
    "hero": None, "timeline": None,
    "hero_beat": HERO, "final_hero": FINAL_HERO, "opening": HOOK,
    "build_up": BUILD_UP, "anticipation": ANTICIPATION, "action": ACTION,
    "climax": CLIMAX, "tail": CELEBRATION, "secondary": SECONDARY,
    "reaction": REACTION,
}


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def _smooth(values: list[float], window: int) -> list[float]:
    """Lissage par moyenne glissante centrée (déterministe, sans aléa)."""
    if not values:
        return []
    w = max(1, int(window))
    half = w // 2
    n = len(values)
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def hero_speed_profile(frames: list[Any]) -> list[dict[str, Any]]:
    """Profil de vitesse RÉEL du héros : ``[{'frame', 'speed'}, ...]``.

    ``speed`` = déplacement réel du centre de bbox entre deux frames suivies
    consécutives (px/frame), lissé par moyenne glissante. Les frames sans suivi
    ne produisent aucun point : rien n'est extrapolé.
    """
    centers = track_frame_centers(frames)
    keys = sorted(centers)
    raw: list[float] = []
    prof: list[int] = []
    for a, b in zip(keys, keys[1:]):
        ca, cb = centers[a], centers[b]
        raw.append(math.hypot(cb[0] - ca[0], cb[1] - ca[1]))
        prof.append(int(b))
    sm = _smooth(raw, SPEED_SMOOTH_WINDOW)
    return [{"frame": f, "speed": round_scale(s, 4)} for f, s in zip(prof, sm)]


def hero_beats(profile: list[dict[str, Any]]) -> dict[str, Any]:
    """Détecte les beats RÉELS du héros (pics de vitesse).

    Renvoie ``primary`` (pic global) et ``secondary`` (second pic significatif,
    suffisamment séparé) — ou ``None`` si la preuve est insuffisante.
    """
    if not profile:
        return {"primary": None, "secondary": None, "peak_speed": 0.0,
                "insufficient_evidence": True}
    speeds = [p["speed"] for p in profile]
    smax = max(speeds)
    if smax <= 1e-6:
        return {"primary": None, "secondary": None, "peak_speed": 0.0,
                "insufficient_evidence": True}
    pi = speeds.index(smax)      # premier maximum => déterminisme
    primary = int(profile[pi]["frame"])

    cutoff = primary + SECOND_BURST_MIN_SEP
    best_i, best_v = None, 0.0
    for i, p in enumerate(profile):
        if p["frame"] < cutoff or p["speed"] < SECOND_BURST_MIN_RATIO * smax:
            continue
        prev_s = profile[i - 1]["speed"] if i > 0 else 0.0
        next_s = profile[i + 1]["speed"] if i + 1 < len(profile) else 0.0
        if p["speed"] >= prev_s and p["speed"] >= next_s and p["speed"] > best_v:
            best_i, best_v = i, p["speed"]
    secondary = int(profile[best_i]["frame"]) if best_i is not None else None
    return {
        "primary": primary,
        "secondary": secondary,
        "peak_speed": round_scale(smax, 4),
        "insufficient_evidence": False,
    }


def hero_start_naive(hero_first: int, hero_peak: int) -> int:
    """Repli documenté : ``HERO_PRE`` frames de montée avant le beat héros."""
    return int(max(hero_first, int(hero_peak) - HERO_PRE))


def hero_action_onset(profile: list[dict[str, Any]], peak_frame: int,
                      ratio: float = 0.35) -> int | None:
    """Frame réelle où démarre la montée d'action menant au beat héros.

    Définition déterministe : dernière frame suivie **avant** ``peak_frame`` dont
    la vitesse lissée reste sous ``ratio × pic``. C'est une mesure sur le profil
    réel du héros — aucun seuil posé « au feeling » sur un événement inventé.
    """
    if not profile or peak_frame is None:
        return None
    smax = max(p["speed"] for p in profile)
    if smax <= 1e-6:
        return None
    thresh = ratio * smax
    onset = None
    for p in profile:
        if p["frame"] >= int(peak_frame):
            break
        if p["speed"] < thresh:
            onset = int(p["frame"])
    return onset


def dominant_track_at(tracks: dict[Any, list], frame: int,
                      prefer_person_sized: bool = True) -> int | None:
    """Trajectoire réelle dominante (plus grande aire de bbox) à ``frame``.

    Déterministe : égalité tranchée par ``track_id`` décroissant. Renvoie
    ``None`` si aucune trajectoire réelle n'existe à cette frame.

    Par défaut on **préfère les trajectoires taille-personne** (le sujet cadré
    est un joueur) ; c'est un choix de cadrage, pas une identité revendiquée.
    """
    pool = tracks or {}
    if prefer_person_sized:
        persons = {tid: tracks[tid] for tid in tracks
                   if is_person_sized(tracks[tid])}
        if persons:
            pool = persons
    best_id, best_area = None, -1.0
    for tid in sorted(pool, key=lambda k: int(k)):
        for item in pool[tid]:
            if not isinstance(item, dict) or int(item.get("frame", -1)) != int(frame):
                continue
            b = item.get("bbox")
            if b is None:
                break
            try:
                area = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
            except (TypeError, ValueError, IndexError):
                break
            if area > best_area:
                best_id, best_area = int(tid), area
            break
    return best_id


def track_targets(tracks: dict[Any, list], track_id: int | None,
                  start_frame: int, end_frame: int,
                  identity: dict[str, Any] | None = None,
                  canonical_id: int | None = None) -> list[dict[str, Any]]:
    """Cibles caméra RÉELLES d'une trajectoire sur ``[start, end]``.

    Chaque cible reprend le centre de bbox réellement suivi ; les frames sans
    suivi **maintiennent la dernière position réelle connue** (jamais une
    position inventée). Renvoie ``[]`` si aucune preuve.

    Quand ``identity`` (Subject Continuity V2.1) est fourni, chaque cible porte
    aussi ``fragment`` = fragment de suivi RÉEL actif à cette frame, et
    ``identity`` = ``canonical_id`` : la caméra reste couplée au fragment actif
    de la **même** identité canonique (aucun id fabriqué).
    """
    if track_id is None or int(track_id) not in {int(k) for k in (tracks or {})}:
        return []
    centers = track_frame_centers(tracks[int(track_id)])
    if not centers:
        return []
    keys = sorted(centers)

    def hold(idx: int) -> tuple[float, float]:
        cand = None
        for k in keys:
            if k <= idx:
                cand = centers[k]
            else:
                break
        return centers[keys[0]] if cand is None else cand

    out: list[dict[str, Any]] = []
    for f in range(int(start_frame), int(end_frame) + 1):
        if f in centers:
            cx, cy = centers[f]
            held = False
        else:
            cx, cy = hold(f)
            held = True
        tgt: dict[str, Any] = {"frame": int(f), "cx": round_scale(cx, 3),
                               "cy": round_scale(cy, 3), "track_id": int(track_id),
                               "held": bool(held)}
        if identity is not None and canonical_id is not None:
            frag = active_track_at(identity, f)
            tgt["fragment"] = int(frag) if frag is not None else int(track_id)
            tgt["identity"] = int(canonical_id)
        out.append(tgt)
    return out


def _active_fragment_at(identity: dict[str, Any] | None, track_id: int | None,
                        frame: int) -> int | None:
    """Fragment de suivi RÉEL actif d'une identité à ``frame`` (fallback ``track_id``)."""
    if identity is not None:
        frag = active_track_at(identity, frame)
        if frag is not None:
            return int(frag)
    return int(track_id) if track_id is not None else None


def _clamp_crop(value: float) -> float:
    return float(min(CROP_FRACTION_MAX, max(CROP_FRACTION_MIN, value)))


def _phase_shot(phase: str, start: int, end: int, tracks: dict[Any, list],
                hero_track: int | None, hero_event: Any, fps: float,
                rife_available: bool, motion_interp: str,
                real_cuts: list[int], index: int, prev_phase: str | None,
                is_last_phase: bool, boundary: str,
                hero_evidence: dict[str, Any],
                hero_lineage_identity: dict[str, Any] | None = None,
                canonical_id: int | None = None) -> Shot | None:
    """Construit un :class:`Shot` pour une phase sur ``[start, end]``."""
    n = end - start + 1
    if n < MIN_SHOT_FRAMES:
        return None

    framing = PHASE_FRAMING.get(phase, {"shot": SHOT_WIDE,
                                       "camera": CAM_STABLE_FOLLOW})
    shot_type = framing["shot"]
    camera = framing["camera"]

    base_crop = CROP_FRACTION.get(shot_type, 1.0)
    delta = CAMERA_CROP_DELTA.get(camera, 0.0)
    if camera == CAM_PULL_BACK:
        crop_start, crop_end = _clamp_crop(base_crop + delta), _clamp_crop(base_crop)
    else:
        crop_start, crop_end = _clamp_crop(base_crop), _clamp_crop(base_crop + delta)

    # Sujet suivi : le héros pour les phases héros ; sinon la trajectoire réelle
    # dominante à la première frame du plan (jamais inventée).
    hero_centric = phase in HERO_CENTRIC_PHASES
    hero_known = hero_track is not None and hero_track in {int(k) for k in tracks}
    if hero_centric and hero_known:
        selected: int | None = int(hero_track)
    else:
        selected = dominant_track_at(tracks, start)
    targets = track_targets(tracks, selected, start, end,
                            identity=(hero_lineage_identity if hero_centric else None),
                            canonical_id=canonical_id)

    # Identité visée : identité canonique héros pour les plans centrés héros,
    # fragment de suivi brut sinon (distinction explicite exigée).
    if hero_centric and hero_known and hero_lineage_identity is not None:
        target_identity = "CANONICAL_HERO_IDENTITY"
        canonical_identity_id: int | None = canonical_id
        active_fragment = _active_fragment_at(hero_lineage_identity, selected, start)
    else:
        target_identity = "RAW_TRACK"
        canonical_identity_id = None
        active_fragment = selected

    event_supported = phase in EVENT_SPEED_PHASES and bool(hero_event)
    curve = speed_curve(phase, event_supported=event_supported,
                        confidence=float(hero_evidence.get("confidence", 0.0)),
                        rife_available=rife_available,
                        motion_interpolation=motion_interp)

    t_in, t_out, t_reason = resolve_transitions(
        prev_phase, phase, start, is_first=(index == 0), is_last=is_last_phase,
        real_cuts=real_cuts)

    evidence = [f"frames:{start}-{end}", f"source_frames:{n}"]
    if hero_event:
        evidence.append(f"hero_event:{hero_event.get('event_id')}")
    if selected is not None:
        evidence.append(f"track:{selected}")
    if hero_centric:
        evidence.append("hero_centric:true")
    if target_identity == "CANONICAL_HERO_IDENTITY":
        evidence.append(f"identity:{canonical_identity_id}")
        if active_fragment is not None and active_fragment != canonical_identity_id:
            evidence.append(f"active_fragment:{active_fragment}")
    if boundary == "REAL_SOURCE_CUT":
        evidence.append("source_cut_boundary")
    evidence.append(f"speed:{curve.behavior}")

    reason = f"{phase} on tracked evidence; transition={t_reason}"
    if hero_centric and not hero_known:
        reason += "; hero track unavailable -> real dominant track used"
    if target_identity == "CANONICAL_HERO_IDENTITY" and active_fragment is not None \
            and active_fragment != canonical_identity_id:
        reason += f"; follows canonical hero via active fragment {active_fragment}"
    if selected is None:
        reason += "; no tracked subject at shot start (wide-safe framing)"

    return Shot(
        shot_id=f"SHOT_{index + 1:03d}",
        phase=phase,
        shot_type=shot_type,
        start_frame=int(start),
        end_frame=int(end),
        start_time_s=round_scale(start / fps, 4),
        end_time_s=round_scale((end + 1) / fps, 4),
        duration_s=round_scale(n / fps, 4),
        duration_frames=int(n),
        selected_track=(int(selected) if selected is not None else None),
        camera_behavior=camera,
        transition_in=t_in,
        transition_out=t_out,
        crop_start=round_scale(crop_start, 4),
        crop_end=round_scale(crop_end, 4),
        crop_easing=("ease_in_out" if camera in (CAM_PUSH_IN, CAM_PULL_BACK)
                     else "linear"),
        subject_y_bias=round_scale(SUBJECT_Y_BIAS.get(shot_type, 0.55), 4),
        speed_curve=curve.to_dict(),
        priority=PHASE_PRIORITY.get(phase, 10),
        evidence=evidence,
        confidence=round_scale(float(hero_evidence.get("confidence", 0.0))
                               if hero_centric else 0.0),
        reason=reason,
        fallback=curve.fallback,
        boundary=boundary,
        target_identity=target_identity,
        canonical_identity_id=canonical_identity_id,
        active_fragment=(int(active_fragment) if active_fragment is not None else None),
        targets=targets,
    )


def _segments_to_shots(segments: list[tuple[int, int, str]],
                       tracks: dict[Any, list], hero_track: int | None,
                       hero_event: Any, fps: float, rife_available: bool,
                       motion_interp: str, real_cuts: list[int],
                       hero_evidence: dict[str, Any],
                       hero_lineage_identity: dict[str, Any] | None = None,
                       canonical_id: int | None = None) -> list[Shot]:
    """Transforme des segments ``(start, end, phase)`` en :class:`Shot` ordonnés."""
    shots: list[Shot] = []
    prev_phase: str | None = None
    for i, (s, e, ph) in enumerate(segments):
        # Une coupe RÉELLE de la source tombant dans le plan est enregistrée :
        # c'est une frontière de plan mesurée, jamais supposée.
        boundary = ("REAL_SOURCE_CUT"
                    if any(int(s) <= int(c) <= int(e) for c in real_cuts)
                    else "CONTINUOUS")
        shot = _phase_shot(ph, s, e, tracks, hero_track, hero_event, fps,
                           rife_available, motion_interp, real_cuts, len(shots),
                           prev_phase, is_last_phase=(i == len(segments) - 1),
                           boundary=boundary, hero_evidence=hero_evidence,
                           hero_lineage_identity=hero_lineage_identity,
                           canonical_id=canonical_id)
        if shot is None:
            continue
        shots.append(shot)
        prev_phase = shot.phase
    if shots:
        shots[-1].transition_out = "FADE_TO_BLACK"
    return shots


# ---------------------------------------------------------------------------
# Résolution déterministe des recouvrements -> partition complète du temps
# ---------------------------------------------------------------------------
def resolve_timeline(candidates: list[tuple[int, int, str]], total_frames: int,
                     min_frames: int = MIN_SHOT_FRAMES) -> list[tuple[int, int, str]]:
    """Convertit des segments candidats en une **partition contiguë** du temps.

    Algorithme (déterministe) :

      1. chaque frame est attribuée au candidat de **plus haute priorité** qui la
         contient (priorités totales) ;
      2. les frames non couvertes héritent du propriétaire précédent (ou du
         premier candidat) — aucune frame n'est perdue, aucun plan n'est inventé ;
      3. les runs plus courts que ``min_frames`` sont fusionnés dans le voisin de
         plus haute priorité (à défaut, le précédent) ;
      4. les runs adjacents de même phase sont fusionnés.
    """
    if total_frames <= 0 or not candidates:
        return []
    owner = [-1] * total_frames
    for idx, (s, e, ph) in enumerate(candidates):
        pr = PHASE_PRIORITY.get(ph, 10)
        lo, hi = max(0, int(s)), min(total_frames - 1, int(e))
        for f in range(lo, hi + 1):
            cur = owner[f]
            if cur < 0:
                owner[f] = idx
                continue
            cur_pr = PHASE_PRIORITY.get(candidates[cur][2], 10)
            if pr > cur_pr or (pr == cur_pr and idx < cur):
                owner[f] = idx

    # 2) frames non couvertes : héritage du précédent (ou du premier candidat).
    first_take = next((o for o in owner if o >= 0), None)
    if first_take is None:
        return []
    for f in range(total_frames):
        if owner[f] < 0:
            owner[f] = owner[f - 1] if f > 0 else first_take

    runs: list[list[int]] = []
    for f in range(total_frames):
        if runs and runs[-1][2] == owner[f]:
            runs[-1][1] = f
        else:
            runs.append([f, f, owner[f]])

    # 3) fusion des runs trop courts.
    while len(runs) > 1:
        short = next((i for i, r in enumerate(runs)
                      if r[1] - r[0] + 1 < min_frames), None)
        if short is None:
            break
        if short == 0:
            tgt = 1
        elif short == len(runs) - 1:
            tgt = short - 1
        else:
            pr_l = PHASE_PRIORITY.get(candidates[runs[short - 1][2]][2], 10)
            pr_r = PHASE_PRIORITY.get(candidates[runs[short + 1][2]][2], 10)
            tgt = (short - 1) if pr_l >= pr_r else (short + 1)
        lo = min(runs[tgt][0], runs[short][0])
        hi = max(runs[tgt][1], runs[short][1])
        runs[tgt][0], runs[tgt][1] = lo, hi
        runs.pop(short)

    # 4) fusion des runs adjacents de même phase.
    out: list[tuple[int, int, str]] = []
    for s, e, idx in runs:
        ph = candidates[idx][2]
        if out and out[-1][2] == ph:
            out[-1] = (out[-1][0], e, ph)
        else:
            out.append((s, e, ph))
    return out


# ---------------------------------------------------------------------------
# Budget temporel : préserve la durée totale dans les bornes de vitesse
# ---------------------------------------------------------------------------
def allocate_output_frames(shots: list[Shot], total_frames: int) -> dict[str, Any]:
    """Répartit ``total_frames`` frames de sortie sur les plans (durée préservée).

    Poids d'un plan : ``Σ_frames / facteur_moyen`` (un ralenti « coûte » plus de
    temps de sortie). Une allocation *plus grand reste* bornée garantit
    ``Σ out_frames == total_frames`` tout en maintenant le **facteur effectif** de
    chaque plan dans ``[SPEED_MIN, SPEED_MAX]``.
    """
    if not shots or total_frames <= 0:
        return {"total_out_frames": 0, "shots": {}, "duration_preserved": False}
    src = [max(1, int(s.duration_frames)) for s in shots]
    factors: list[float] = []
    for s in shots:
        c = s.speed_curve or {}
        factors.append(max(SPEED_MIN, min(SPEED_MAX,
                       (float(c.get("start_factor", 1.0)) +
                        float(c.get("end_factor", 1.0))) / 2.0)))
    weights = [n / f for n, f in zip(src, factors)]
    wsum = sum(weights) or 1.0
    ideal = [total_frames * w / wsum for w in weights]

    lo = [int(math.ceil(n / SPEED_MAX)) for n in src]
    hi = [int(math.floor(n / SPEED_MIN)) for n in src]
    alloc = [int(min(hi[i], max(lo[i], math.floor(ideal[i])))) for i in range(len(src))]
    remainder = total_frames - sum(alloc)
    order = sorted(range(len(src)),
                   key=lambda i: (-(ideal[i] - math.floor(ideal[i])), i))
    guard = 0
    while remainder != 0 and guard < 20 * len(src) + 20:
        guard += 1
        progressed = False
        for i in order:
            if remainder == 0:
                break
            if remainder > 0 and alloc[i] < hi[i]:
                alloc[i] += 1
                remainder -= 1
                progressed = True
            elif remainder < 0 and alloc[i] > lo[i]:
                alloc[i] -= 1
                remainder += 1
                progressed = True
        if not progressed:
            break

    out: dict[str, Any] = {"total_out_frames": 0, "shots": {}, "src_total_frames": total_frames}
    total = 0
    for s, n, a in zip(shots, src, alloc):
        eff = n / float(a) if a else 1.0
        out["shots"][s.shot_id] = {
            "src_frames": int(n),
            "out_frames": int(a),
            "effective_factor": round_scale(eff),
            "in_bounds": bool(SPEED_MIN - 1e-6 <= eff <= SPEED_MAX + 1e-6),
        }
        total += a
    out["total_out_frames"] = int(total)
    out["duration_preserved"] = bool(total == total_frames)
    return out


# ---------------------------------------------------------------------------
# Candidats -> plan
# ---------------------------------------------------------------------------
def _absent(phase: str, reason: str, detail: str = "") -> dict[str, Any]:
    return {"phase": phase, "reason": reason, "detail": detail}


def _phase_of_fallback(f: dict[str, Any]) -> str | None:
    return _FALLBACK_PHASE.get(f.get("stage"))


def _candidates(hero_first: int, hero_peak: int, hero_last: int,
                total_frames: int, scene: dict[str, Any],
                fallbacks: list[dict[str, Any]],
                speed_profile: list[dict[str, Any]] | None = None
                ) -> list[tuple[int, int, str]]:
    """Construit les segments candidats (phases) à partir de la preuve héros."""
    cands: list[tuple[int, int, str]] = []

    # Démarrage d'action RÉEL : montée de vitesse du héros avant son beat.
    onset = hero_action_onset(speed_profile or [], hero_peak)
    onset = int(max(hero_first, onset)) if onset is not None else hero_start_naive(
        hero_first, hero_peak)
    if onset >= hero_peak:
        onset = int(max(hero_first, hero_peak - HERO_PRE))

    hero_start = int(onset)
    hero_end = int(min(hero_last, hero_peak + HERO_POST))
    if hero_end - hero_start + 1 < MIN_SHOT_FRAMES:
        hero_end = int(min(hero_last, hero_start + MIN_SHOT_FRAMES - 1))
    cands.append((hero_start, hero_end, HERO))

    # Anticipation / action : fenêtres réelles avant la montée d'action.
    if hero_start - 1 >= hero_first:
        ant_hi = hero_start - 1
        ant_lo = max(hero_first, hero_start - ANTICIPATION_LEN)
        cands.append((ant_lo, ant_hi, ANTICIPATION))
        if ant_lo - 1 >= hero_first:
            act_lo = max(hero_first, ant_lo - ACTION_LEN)
            if ant_lo - act_lo >= MIN_SHOT_FRAMES:
                cands.append((act_lo, ant_lo - 1, ACTION))
            else:
                fallbacks.append({"stage": "action", "reason": "ACTION_WINDOW_TOO_SHORT",
                                  "detail": f"fenêtre {act_lo}-{ant_lo - 1}"})
            # Build-up : frames réelles entre l'ouverture et l'action.
            bu_hi = act_lo - 1
            bu_lo = min(HOOK_FRAMES, hero_first)
            if bu_hi - bu_lo + 1 >= MIN_SHOT_FRAMES:
                cands.append((bu_lo, bu_hi, BUILD_UP))
        elif hero_start - hero_first < ANTICIPATION_LEN:
            fallbacks.append({"stage": "anticipation",
                              "reason": "SHORT_PRE_HERO_WINDOW",
                              "detail": f"{hero_start - hero_first} frames avant le héros"})

    # Ouverture : HOOK sur les premières frames réelles.
    pre_len = hero_first
    if pre_len >= MIN_SHOT_FRAMES:
        hook_end = min(HOOK_FRAMES, pre_len) - 1
        if hook_end + 1 >= MIN_SHOT_FRAMES:
            cands.append((0, hook_end, HOOK))
        else:
            cands.append((0, pre_len - 1, HOOK))
    else:
        fallbacks.append({"stage": "opening", "reason": "NO_PRE_HERO_FRAMES",
                          "detail": f"héros suivi dès la frame {hero_first}"})

    # Réaction : continuation réelle de la trajectoire héros.
    reaction_start = hero_end + 1
    reaction_end = min(hero_last, reaction_start + REACTION_LEN - 1)
    if reaction_end >= reaction_start:
        cands.append((reaction_start, reaction_end, REACTION))
    else:
        fallbacks.append({"stage": "reaction", "reason": "NO_POST_HERO_FRAMES",
                          "detail": f"trajectoire héros terminée à la frame {hero_last}"})

    # Final hero : second beat réel (retour sur le sujet) — sinon omis.
    beats_secondary = scene.get("_hero_secondary")
    if beats_secondary is not None:
        sec = int(beats_secondary)
        fin_lo = int(max(reaction_end + 1, sec - FINAL_HERO_PRE))
        fin_hi = int(min(hero_last, sec + FINAL_HERO_POST))
        if fin_hi - fin_lo + 1 >= MIN_SHOT_FRAMES:
            cands.append((fin_lo, fin_hi, FINAL_HERO))
        else:
            fallbacks.append({"stage": "final_hero", "reason": "SECOND_BEAT_TOO_SHORT",
                              "detail": f"second beat {fin_lo}-{fin_hi}"})
    else:
        fallbacks.append({"stage": "final_hero", "reason": "NO_SECOND_HERO_BEAT",
                          "detail": "aucun second pic réel de vitesse du héros"})

    # Queue : célébration réelle + climax (pic de mouvement réel) + outro.
    tail_start = int(max(hero_end, reaction_end) + 1)
    if total_frames - tail_start >= MIN_SHOT_FRAMES:
        outro_start = max(tail_start, total_frames - OUTRO_FRAMES)
        mid_end = outro_start - 1
        c_lo = tail_start
        c_hi = mid_end
        climax_peak = scene.get("motion_peak_frame")
        climax_used = False
        if climax_peak is not None and tail_start <= int(climax_peak) <= mid_end:
            cl = max(tail_start, int(climax_peak) - CLIMAX_HALF)
            ch = min(mid_end, int(climax_peak) + CLIMAX_HALF)
            if ch - cl + 1 >= MIN_SHOT_FRAMES:
                if cl - tail_start >= MIN_SHOT_FRAMES:
                    cands.append((tail_start, cl - 1, CELEBRATION))
                    c_lo = cl
                else:
                    c_lo = tail_start
                c_hi = ch
                cands.append((c_lo, c_hi, CLIMAX))
                climax_used = True
                if mid_end - c_hi >= MIN_SHOT_FRAMES:
                    cands.append((c_hi + 1, mid_end, SECONDARY))
                else:
                    fallbacks.append({"stage": "secondary",
                                      "reason": "SECONDARY_WINDOW_TOO_SHORT",
                                      "detail": f"fenêtre {c_hi + 1}-{mid_end}"})
        if not climax_used:
            fallbacks.append({
                "stage": "climax", "reason": "NO_ISOLATED_MOTION_PEAK",
                "detail": ("le pic de mouvement réel de la source n'est pas "
                           "séparable en un plan distinct"),
            })
        if c_lo <= c_hi:
            cands.append((c_lo, c_hi, CELEBRATION))
        cands.append((outro_start, total_frames - 1, OUTRO))
    else:
        cands.append((tail_start, total_frames - 1, OUTRO))
        fallbacks.append({"stage": "tail", "reason": "TAIL_TOO_SHORT_FOR_CELEBRATION",
                          "detail": f"queue {tail_start}-{total_frames - 1}"})
    return cands


def _evidence_summary(evidence: dict[str, Any], scene: dict[str, Any],
                      hero_beats: dict[str, Any] | None = None,
                      hero_presence: dict[str, Any] | None = None,
                      motion: list[float] | None = None) -> dict[str, Any]:
    """Résumé lisible et sérialisable des preuves réellement utilisées."""
    balls = evidence.get("ball", {}) or {}
    tracks = evidence.get("tracks", {}) or {}
    out: dict[str, Any] = {
        "tracks_total": tracks.get("count", 0),
        "person_sized_tracks": tracks.get("person_sized_count", 0),
        "ball_evidence_available": bool(balls.get("available")),
        "ball_detections": balls.get("count", 0),
        "real_cut_frames": list(scene.get("cut_frames", [])),
        "motion_mean": round_scale(scene.get("motion_mean", 0.0), 4),
        "motion_peak_frame": scene.get("motion_peak_frame"),
        "camera_optical_flow": evidence.get("camera_motion", {}),
    }
    if hero_beats is not None:
        out["hero_beats"] = hero_beats
    if hero_presence is not None:
        out["hero_presence"] = hero_presence
    if motion:
        out["motion_profile_frames"] = len(motion)
    return out


# ---------------------------------------------------------------------------
# Subject Continuity V2.1 — helpers d'identité canonique
# ---------------------------------------------------------------------------
def _identity_has_lineage(identity: dict[str, Any] | None) -> bool:
    """Vrai si l'identité canonique regroupe plus d'un fragment brut."""
    return bool(identity) and len(identity.get("source_track_ids", [])) > 1


def _identity_confidence(identity: dict[str, Any] | None) -> float:
    if not identity:
        return 0.0
    return float(identity.get("confidence", 0.0) or 0.0)


def _identity_evidence(identity: dict[str, Any] | None) -> list[str]:
    if not identity:
        return []
    return [str(e) for e in identity.get("evidence", [])]


def resolve_hero_identity(tracks: dict[Any, list] | None,
                          hero_moment: dict[str, Any] | None,
                          continuity: dict[str, Any] | None = None,
                          ) -> tuple[dict[str, Any] | None, dict[Any, list]]:
    """Résout l'identité canonique du héros et les trajectoires de plan.

    Renvoie ``(identity, plan_tracks)`` :

      * ``identity`` : identité canonique héros (lignée) ou ``None`` si le héros
        n'est pas suivi ;
      * ``plan_tracks`` : le dict de trajectoires à consommer par le director.
        Quand la lignée regroupe plusieurs fragments, le héros y est remplacé par
        la **concaténation réelle** de ses fragments (sous son id canonique),
        afin que vitesse/beats/cibles proviennent de la trajectoire continue
        réelle — jamais d'une identité fabriquée.
    """
    tracks = tracks or {}
    hero_moment = hero_moment or {}
    hero_track = hero_moment.get("hero_track")
    if hero_track is None:
        hero_track = hero_moment.get("track_id")
    hero_track = int(hero_track) if hero_track is not None else None
    if hero_track is None:
        return None, dict(tracks)

    if continuity is None:
        continuity = analyze_continuity(tracks, hero_track=hero_track)

    identity = continuity.get("hero_identity")
    if not identity:
        identity = hero_identity(continuity, hero_track, tracks=tracks)

    plan_tracks = dict(tracks)
    if _identity_has_lineage(identity):
        frames = identity_frames(tracks, identity)
        if frames:
            plan_tracks[hero_track] = frames
    return identity, plan_tracks


def build_edit_plan(video: str, tracks: dict[Any, list] | None = None,
                    hero_moment: dict[str, Any] | None = None,
                    events: list[dict] | None = None,
                    ball_detections: list[dict] | None = None,
                    camera_transforms: list[dict] | None = None,
                    width: int = 0, height: int = 0, fps: float = 30.0,
                    total_frames: int = 0, scene: dict[str, Any] | None = None,
                    rife_available: bool = False,
                    motion_interpolation: str = "TEMPORAL_RESAMPLE_NO_RIFE",
                    continuity: dict[str, Any] | None = None,
                    ) -> EditPlan:
    """Construit le plan de montage cinématique — déterministe, piloté par la preuve.

    Toutes les entrées proviennent des sorties réelles du pipeline. Le plan
    résultant partitionne **exactement** ``total_frames`` (durée préservée).

    ``continuity`` (Subject Continuity V2.1) : rapport de continuité de sujet
    (``track_continuity.json``). Quand il regroupe plusieurs fragments bruts en
    une identité canonique héros, le director consomme cette lignée : les beats,
    la vitesse et les cibles caméra proviennent de la trajectoire continue réelle,
    et la caméra suit l'**identité canonique** à travers ses fragments actifs.
    ``continuity=None`` préserve exactement le comportement historique.
    """
    tracks = tracks or {}
    hero_moment = hero_moment or {}
    scene = dict(scene or {})
    events = events or []

    if total_frames <= 0:
        total_frames = int(scene.get("frames") or 0)
    if total_frames <= 0:
        last = -1
        for tid in sorted(tracks, key=lambda k: int(k)):
            for item in tracks[tid]:
                if isinstance(item, dict) and item.get("frame") is not None:
                    last = max(last, int(item["frame"]))
        total_frames = last + 1
    if total_frames <= 0:
        return EditPlan(source_video=str(video), source_width=int(width),
                        source_height=int(height), fps=float(fps),
                        fallbacks=[{"stage": "timeline", "reason": "NO_FRAMES",
                                    "detail": "aucune frame source mesurable"}],
                        phases_absent=[_absent(p, "NO_TIMELINE") for p in PHASE_ORDER])

    hero_track = hero_moment.get("hero_track")
    if hero_track is None:
        hero_track = hero_moment.get("track_id")
    hero_track = int(hero_track) if hero_track is not None else None
    hero_event = hero_moment.get("event") or None
    hero_conf = float(hero_moment.get("score") or 0.0)

    # --- Subject Continuity V2.1 : identité canonique + trajectoires de plan --
    hero_ident, plan_tracks = resolve_hero_identity(tracks, hero_moment, continuity)
    has_lineage = _identity_has_lineage(hero_ident)
    continuity_links = list((hero_ident or {}).get("links", [])) if hero_ident else []
    continuity_conf = _identity_confidence(hero_ident)
    continuity_evid = _identity_evidence(hero_ident)
    hero_lineage = list((hero_ident or {}).get("source_track_ids", [])) if hero_ident else []
    camera_follow_mode = ("HERO_IDENTITY" if has_lineage
                          else ("RAW_TRACK" if hero_track is not None else "NONE"))

    evidence = build_evidence(tracks=plan_tracks, ball_detections=ball_detections,
                              camera_transforms=camera_transforms, width=width,
                              height=height, total_frames=total_frames, scene=scene)
    motion = scene.get("motion")
    real_cuts = list(scene.get("cut_frames", []))

    presence = hero_presence(hero_track, plan_tracks)
    hero_frames = plan_tracks.get(hero_track, []) if hero_track is not None else []
    hero_supported = bool(presence.get("present")) and is_person_sized(hero_frames)

    # --- Cas sans héros réellement suivi : plan minimal honnête --------------
    if not hero_supported:
        fallbacks = [{
            "stage": "hero",
            "reason": ("NO_HERO_TRACK" if not presence.get("present")
                       else "HERO_TRACK_NOT_PERSON_SIZED"),
            "detail": ("aucune trajectoire « taille personne » pour le sujet "
                       "héros : le directeur ne fabrique aucun beat centré héros"),
            "hero_track": hero_track,
        }]
        if total_frames > HOOK_FRAMES + MIN_SHOT_FRAMES:
            segments = [(0, HOOK_FRAMES - 1, HOOK),
                        (HOOK_FRAMES, total_frames - 1, OUTRO)]
        else:
            segments = [(0, total_frames - 1, HOOK)]
        shots = _segments_to_shots(
            segments, tracks=plan_tracks, hero_track=None, hero_event=None, fps=fps,
            rife_available=rife_available, motion_interp=motion_interpolation,
            real_cuts=real_cuts, hero_evidence={"confidence": 0.0})
        phases_present = sorted({s.phase for s in shots},
                               key=lambda p: PHASE_ORDER.index(p))
        plan = EditPlan(
            source_video=str(video), source_width=int(width),
            source_height=int(height), fps=float(fps), total_frames=int(total_frames),
            total_duration_s=round_scale(total_frames / float(fps), 4),
            hero_track=hero_track, hero_event_id=None,
            hero_method=hero_moment.get("method"), followed_track=None,
            hero_camera_contract=False, fallbacks=fallbacks,
            evidence_used=_evidence_summary(evidence, scene),
            phases_present=phases_present,
            phases_absent=[_absent(p, "NO_HERO_EVIDENCE") for p in PHASE_ORDER
                           if p not in phases_present],
            shots=shots,
            hero_identity=(hero_ident or {}),
            hero_lineage=hero_lineage,
            continuity_links=continuity_links,
            continuity_confidence=continuity_conf,
            continuity_evidence=continuity_evid,
            camera_follow_mode="NONE")
        plan.time_map = allocate_output_frames(shots, total_frames)
        plan.duration_preserved = bool(plan.time_map.get("duration_preserved"))
        plan.claims = {
            "model_truth_unchanged": True, "rife_used": bool(rife_available),
            "interpolation_mode": ("rife" if rife_available else motion_interpolation),
            "novel_view": "NOT_USED", "generated_pixels": False,
            "real_source_only": True,
        }
        return plan

    # --- Héros soutenu : structure cinématique pilotée par la preuve ---------
    hero_first = int(presence["first_frame"])
    hero_last = int(presence["last_frame"])
    profile = hero_speed_profile(hero_frames)
    beats = hero_beats(profile)
    hero_peak = beats["primary"]
    if hero_peak is None:
        hero_peak = (hero_event or {}).get("peak_frame")
        if hero_peak is None:
            hero_peak = (hero_first + hero_last) // 2
            fallbacks_note = "HERO_PEAK_UNDEFINED"
        else:
            fallbacks_note = "HERO_PEAK_FROM_EVENT"
    else:
        fallbacks_note = None
    hero_peak = int(max(hero_first, min(hero_last, int(hero_peak))))

    scene_for_candidates = dict(scene)
    scene_for_candidates["_hero_secondary"] = beats.get("secondary")

    fallbacks: list[dict[str, Any]] = []
    if fallbacks_note:
        fallbacks.append({"stage": "hero_beat", "reason": fallbacks_note,
                          "detail": f"beat héros = {hero_peak}"})
    if beats.get("secondary") is None and not beats.get("insufficient_evidence"):
        fallbacks.append({"stage": "final_hero", "reason": "HERO_NOT_TRACKED_IN_FINAL_WINDOW",
                          "detail": f"trajectoire héros terminée à la frame {hero_last}"})
    # Subject Continuity V2.1 : si une continuation proche du héros a été
    # REJETÉE faute de preuve suffisante, on l'expose honnêtement (jamais un
    # fragment fusionné de force).
    if hero_ident and hero_ident.get("fallback"):
        fallbacks.append({
            "stage": "continuity",
            "reason": str(hero_ident.get("fallback")),
            "detail": ("aucune continuation crédible du héros : fragment de suivi "
                       "laissé non résolu (pas de merge fabriqué)"),
            "target_track_id": (hero_ident.get("fallback_detail") or {}).get("target_track_id"),
        })

    candidates = _candidates(hero_first, hero_peak, hero_last, total_frames,
                             scene_for_candidates, fallbacks, speed_profile=profile)
    segments = resolve_timeline(candidates, total_frames)
    if not segments:
        fallbacks.append({"stage": "timeline", "reason": "NO_SEGMENTS",
                          "detail": "aucun segment candidat résolu"})
        segments = [(0, total_frames - 1, HOOK)]

    shots = _segments_to_shots(segments, tracks=plan_tracks, hero_track=hero_track,
                              hero_event=hero_event, fps=fps,
                              rife_available=rife_available,
                              motion_interp=motion_interpolation,
                              real_cuts=real_cuts,
                              hero_evidence={"confidence": hero_conf},
                              hero_lineage_identity=hero_ident,
                              canonical_id=hero_track)

    phases_present = sorted({s.phase for s in shots},
                            key=lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else 99)
    phases_absent: list[dict[str, Any]] = []
    for p in PHASE_ORDER:
        if p in phases_present:
            continue
        matched = [f for f in fallbacks if _phase_of_fallback(f) == p]
        if matched:
            for f in matched:
                phases_absent.append(_absent(p, f["reason"], f.get("detail", "")))
        else:
            phases_absent.append(_absent(p, "INSUFFICIENT_EVIDENCE"))

    followed_track = hero_track
    hero_camera_contract = bool(
        hero_track is not None
        and followed_track == hero_track
        and all(s.selected_track == hero_track
                for s in shots if s.phase in HERO_CENTRIC_PHASES))

    plan = EditPlan(
        source_video=str(video),
        source_width=int(width), source_height=int(height), fps=float(fps),
        total_frames=int(total_frames),
        total_duration_s=round_scale(total_frames / float(fps), 4),
        hero_track=hero_track,
        hero_event_id=(hero_event or {}).get("event_id"),
        hero_method=hero_moment.get("method"),
        followed_track=followed_track,
        hero_camera_contract=hero_camera_contract,
        fallbacks=fallbacks,
        evidence_used=_evidence_summary(evidence, scene, hero_beats=beats,
                                        hero_presence=presence, motion=motion),
        phases_present=phases_present,
        phases_absent=phases_absent,
        shots=shots,
        hero_identity=(hero_ident or {}),
        hero_lineage=hero_lineage,
        continuity_links=continuity_links,
        continuity_confidence=continuity_conf,
        continuity_evidence=continuity_evid,
        camera_follow_mode=camera_follow_mode,
    )
    # Preuve de continuité consignée (identité canonique, lignée, contrat caméra).
    plan.evidence_used["hero_identity"] = {
        "canonical_track_id": (hero_ident or {}).get("canonical_track_id"),
        "source_track_ids": hero_lineage,
        "fragments": len(hero_lineage),
        "confidence": continuity_conf,
        "fallback": (hero_ident or {}).get("fallback"),
        "camera_follow_mode": camera_follow_mode,
        "has_lineage": bool(has_lineage),
    }
    plan.time_map = allocate_output_frames(shots, total_frames)
    plan.duration_preserved = bool(plan.time_map.get("duration_preserved"))
    plan.claims = {
        "model_truth_unchanged": True,
        "rife_used": bool(rife_available),
        "interpolation_mode": ("rife" if rife_available else motion_interpolation),
        "novel_view": "NOT_USED",
        "generated_pixels": False,
        "real_source_only": True,
    }
    return plan
