"""Subject Continuity V2.1 — identité canonique à partir de fragments de suivi.

Rôle
----
Le suivi réel (ByteTrack sur détections YOLO) **fragmente** naturellement une
même personne en plusieurs ``track_id`` lorsque surviennent une occlusion, un
raté du détecteur, un croisement de joueurs ou une brève disparition. Le sujet
« héros » se retrouve alors tronqué (ex. le track 9 s'arrête à la frame 361
alors que la source en contient 484), ce qui empêche toute continuation fiable
(REACTION / FINAL_HERO / SECONDARY).

Cette couche **n'ajoute aucun nouveau traqueur** et **ne fabrique aucune
identité**. Elle relie, de façon *déterministe*, des fragments de suivi bruts
déjà mesurés en une **identité canonique** lorsque — et seulement lorsque —
des preuves mesurées suffisent.

    tracks bruts
      → analyse de continuité
      → identité canonique (lignée)
      → Director V2
      → caméra consciente du fragment actif
      → rendu
      → QC

Politique de vérité
-------------------
  * Aucune preuve faible isolée ne suffit à fusionner deux fragments.
  * Un lien accepté/rejeté est **toujours** explicable : ``source_track_id``,
    ``target_track_id``, ``score``, ``confidence``, termes de preuve, ``decision``
    et ``rejection_reason`` quand rejeté.
  * Les identités brutes ne sont **jamais** modifiées (aucune réécriture des
    résultats de suivi historiques) : on distingue explicitement
    ``RAW TRACK`` et ``CANONICAL IDENTITY``.
  * Preuve manquante (OSNet indisponible, SAM2 indisponible) → l'analyse
    continue **sans simulation** : la preuve est marquée ``unavailable`` et les
    poids sont renormalisés. Le pipeline échoue « en douceur ».
  * Preuve contradictoire (saut spatial + vitesse incohérente, apparence
    contradictoire) → **PAS DE FUSION**, avec ``fallback`` explicite
    ``UNRESOLVED_TRACK_FRAGMENT``.

Déterminisme
------------
Toutes les itérations sur dictionnaires sont triées ; le départage des liens est
total et stable (``-score``, puis ``source_track_id``, puis ``target_track_id``) ;
les flottants sont arrondis de façon reproductible. Pour une même preuve, le
résultat est identique — indépendamment de l'ordre d'insertion.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA = "goalreel.track_continuity.v1"
POLICY = "NO_MERGE_WITHOUT_EVIDENCE"
FALLBACK_UNRESOLVED = "UNRESOLVED_TRACK_FRAGMENT"
FALLBACK_NO_APPEARANCE = "APPEARANCE_EVIDENCE_UNAVAILABLE"
FALLBACK_NO_SAM2 = "SAM2_EVIDENCE_UNAVAILABLE"

# ---------------------------------------------------------------------------
# Poids (documentés). ``CORE_WEIGHTS`` somme à 1.0 sans preuve apparence.
# La preuve d'apparence (OSNet) et la preuve ballon optionnelle sont ajoutées
# **puis renormalisées** uniquement quand elles sont réellement disponibles.
# ---------------------------------------------------------------------------
CORE_WEIGHTS: dict[str, float] = {
    "temporal": 0.26,     # proximité temporelle (taille du trou)
    "spatial": 0.26,      # continuité de position (prédiction vs réel)
    "velocity": 0.20,     # continuité de vitesse (direction + amplitude)
    "geometry": 0.14,     # continuité de taille / ratio de bbox
    "persistence": 0.14,  # support en frames des deux fragments
}
APPEARANCE_WEIGHT = 0.22      # OSNet cosine (uniquement si embeddings réels)
OPTIONAL_EVENT_WEIGHT = 0.08  # proximité ballon réelle (uniquement si détectée)

# ---------------------------------------------------------------------------
# Seuils / bornes (documentés).
# ---------------------------------------------------------------------------
MAX_GAP_FRAMES = 45           # 1.5 s @30fps — aucun pontage arbitrairement long
MIN_GAP_FRAMES = 1            # un trou nul (chevauchement) n'est pas une suite
ACCEPT_THRESHOLD = 0.55       # score minimal pour accepter une fusion
APPEARANCE_MIN_COS = 0.20     # cosine en dessous duquel l'apparence CONTREDIT
SPATIAL_REJECT_PX = 260.0     # saut de position absolu rédhibitoire (px)
SPATIAL_SOFT_PX = 150.0       # saut modéré qui, combiné à une vitesse incohérente, rejette
VELOCITY_REJECT = 14.0        # incohérence de vitesse (px/frame) rédhibitoire

# ---------------------------------------------------------------------------
# Échelles de saturation (constantes, documentées).
# ---------------------------------------------------------------------------
TEMPORAL_SCALE = 24.0         # frames
SPATIAL_SCALE = 120.0         # px
VELOCITY_SCALE = 8.0          # px/frame
GEOMETRY_SCALE = 40.0         # px (largeur/hauteur)
PERSISTENCE_SCALE = 60.0      # frames
BALL_SCALE = 140.0            # px (proximité ballon réel)

EPS = 1e-9
_VEL_WINDOW = 5               # frames utilisées pour estimer une vitesse à une borne


def round_scale(value: float, digits: int = 6) -> float:
    """Arrondi reproductible (stabilité du déterminisme JSON)."""
    return float(round(float(value), digits))


def _sat(value: float, scale: float) -> float:
    """Saturation douce monotone dans ``[0, 1)`` : ``x / (x + scale)``."""
    value = float(value or 0.0)
    if value <= 0.0:
        return 0.0
    return value / (value + float(scale))


def _bbox_of(item: Any):
    if isinstance(item, dict):
        b = item.get("bbox")
    else:
        b = item
    if b is None:
        return None
    try:
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except (TypeError, ValueError, IndexError):
        return None


def _frame_of(item: Any):
    if isinstance(item, dict) and item.get("frame") is not None:
        try:
            return int(item["frame"])
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Fragment de suivi (RAW TRACK) — statistiques factuelles, rien d'inventé.
# ---------------------------------------------------------------------------
@dataclass
class Fragment:
    track_id: int
    frames: list[dict[str, Any]]          # triés par frame (dédupliqués)
    centers: dict[int, tuple[float, float]]
    sizes: dict[int, tuple[float, float]]
    first_frame: int
    last_frame: int
    length: int
    mean_center: tuple[float, float]
    mean_size: tuple[float, float]
    conf_mean: float
    person_sized: bool

    def center_span(self) -> int:
        return self.last_frame - self.first_frame + 1

    def continuity(self) -> float:
        span = self.center_span()
        return min(1.0, self.length / float(span or 1))


def build_fragments(tracks: dict[Any, list], width: float = 0.0,
                    height: float = 0.0) -> dict[int, Fragment]:
    """Construit les :class:`Fragment` factuels depuis ``{track_id: [items]}``.

    Aucune frame n'est inventée : les items sans bbox exploitable sont ignorés.
    Les clés sont triées (déterminisme).
    """
    from ..cinematic.evidence_model import PERSON_MAX_HEIGHT, PERSON_MAX_WIDTH

    frags: dict[int, Fragment] = {}
    for raw_id in sorted((tracks or {}), key=lambda k: int(k)):
        tid = int(raw_id)
        items = []
        for item in (tracks[raw_id] or []):
            if not isinstance(item, dict):
                continue
            f = _frame_of(item)
            b = _bbox_of(item)
            if f is None or b is None:
                continue
            items.append({"frame": int(f), "bbox": list(b),
                          "confidence": float(item.get("confidence", 0.0) or 0.0)})
        # dédup par frame (dernier item gagne, tri stable)
        by_frame: dict[int, dict[str, Any]] = {}
        for it in items:
            by_frame[int(it["frame"])] = it
        ordered = [by_frame[k] for k in sorted(by_frame)]
        if not ordered:
            continue
        centers = {int(it["frame"]): ((it["bbox"][0] + it["bbox"][2]) / 2.0,
                                      (it["bbox"][1] + it["bbox"][3]) / 2.0)
                   for it in ordered}
        sizes = {int(it["frame"]): (it["bbox"][2] - it["bbox"][0],
                                    it["bbox"][3] - it["bbox"][1])
                 for it in ordered}
        frames_sorted = sorted(centers)
        xs = [centers[f][0] for f in frames_sorted]
        ys = [centers[f][1] for f in frames_sorted]
        ws = [sizes[f][0] for f in frames_sorted]
        hs = [sizes[f][1] for f in frames_sorted]
        mean_w = sum(ws) / len(ws)
        mean_h = sum(hs) / len(hs)
        conf = sum(it["confidence"] for it in ordered) / len(ordered)
        frags[tid] = Fragment(
            track_id=tid,
            frames=ordered,
            centers=centers,
            sizes=sizes,
            first_frame=int(frames_sorted[0]),
            last_frame=int(frames_sorted[-1]),
            length=len(frames_sorted),
            mean_center=(sum(xs) / len(xs), sum(ys) / len(ys)),
            mean_size=(mean_w, mean_h),
            conf_mean=float(conf),
            person_sized=bool(mean_w <= PERSON_MAX_WIDTH and mean_h <= PERSON_MAX_HEIGHT),
        )
    return frags


# ---------------------------------------------------------------------------
# Cinématique de borne (vitesse réelle estimée aux extrémités d'un fragment)
# ---------------------------------------------------------------------------
def _velocity_at_end(frag: Fragment, window: int = _VEL_WINDOW) -> tuple[float, float]:
    keys = sorted(frag.centers)
    if len(keys) < 2:
        return (0.0, 0.0)
    hi = keys[-1]
    lo = keys[max(0, len(keys) - 1 - window)]
    if hi == lo:
        return (0.0, 0.0)
    dt = float(hi - lo)
    c_hi, c_lo = frag.centers[hi], frag.centers[lo]
    return ((c_hi[0] - c_lo[0]) / dt, (c_hi[1] - c_lo[1]) / dt)


def _velocity_at_start(frag: Fragment, window: int = _VEL_WINDOW) -> tuple[float, float]:
    keys = sorted(frag.centers)
    if len(keys) < 2:
        return (0.0, 0.0)
    lo = keys[0]
    hi = keys[min(len(keys) - 1, window)]
    if hi == lo:
        return (0.0, 0.0)
    dt = float(hi - lo)
    c_lo, c_hi = frag.centers[lo], frag.centers[hi]
    return ((c_hi[0] - c_lo[0]) / dt, (c_hi[1] - c_lo[1]) / dt)


def _predict_center(frag: Fragment, at_frame: int) -> tuple[float, float]:
    """Position prédite (extrapolation linéaire bornée) du fragment à ``at_frame``."""
    cx, cy = frag.centers[frag.last_frame]
    vx, vy = _velocity_at_end(frag)
    dt = int(at_frame) - int(frag.last_frame)
    # Extrapolation bornée : jamais au-delà de MAX_GAP_FRAMES de portée.
    dt = max(0, min(MAX_GAP_FRAMES, dt))
    return (cx + vx * dt, cy + vy * dt)


# ---------------------------------------------------------------------------
# Apparence (OSNet) — cosine sur embeddings L2-normalisés réels, sinon ``None``.
# ---------------------------------------------------------------------------
def _cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return None
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Score d'un lien candidat (source → target)
# ---------------------------------------------------------------------------
def score_link(a: Fragment, b: Fragment, *, embeddings: dict[int, Any] | None = None,
               ball_centers: dict[int, tuple[float, float]] | None = None,
               ) -> dict[str, Any]:
    """Calcule le score borné d'un lien ``a → b`` + tous ses termes de preuve.

    Ne décide pas : renvoie un dict explicable (``terms``, ``raw``, ``score``).
    La décision (ACCEPTED/REJECTED) est prise par :func:`_decide`.
    """
    gap = int(b.first_frame) - int(a.last_frame)
    pred = _predict_center(a, b.first_frame)
    actual = b.centers[b.first_frame]
    dist = math.hypot(pred[0] - actual[0], pred[1] - actual[1])

    v_pred = _velocity_at_end(a)
    v_target = _velocity_at_start(b)
    vel_delta = math.hypot(v_pred[0] - v_target[0], v_pred[1] - v_target[1])

    sw_a, sh_a = a.mean_size
    sw_b, sh_b = b.mean_size
    size_delta = abs(sw_a - sw_b) + abs(sh_a - sh_b)
    # ratio de forme (aspect) : évite de relier un joueur à un blob large
    ar_a = sw_a / (sh_a + EPS)
    ar_b = sw_b / (sh_b + EPS)
    ar_delta = abs(ar_a - ar_b)

    terms: dict[str, Any] = {}
    terms["temporal"] = {
        "value": _sat(0.0, 0.0) if gap <= 0 else math.exp(-float(gap) / TEMPORAL_SCALE),
        "weight": None, "contribution": None,
    }
    terms["spatial"] = {
        "value": math.exp(-dist / SPATIAL_SCALE),
        "weight": None, "contribution": None,
    }
    terms["velocity"] = {
        "value": math.exp(-vel_delta / VELOCITY_SCALE),
        "weight": None, "contribution": None,
    }
    terms["geometry"] = {
        "value": math.exp(-(size_delta / GEOMETRY_SCALE + ar_delta)),
        "weight": None, "contribution": None,
    }
    terms["persistence"] = {
        "value": _sat(min(a.length, b.length), PERSISTENCE_SCALE),
        "weight": None, "contribution": None,
    }

    cos = _cosine((embeddings or {}).get(a.track_id), (embeddings or {}).get(b.track_id))
    appearance_available = cos is not None
    if appearance_available:
        # mappe cosine [-1,1] -> [0,1] (cosine 0 => 0.5)
        terms["appearance"] = {"value": max(0.0, min(1.0, (cos + 1.0) / 2.0)),
                               "available": True, "weight": None, "contribution": None}
    else:
        terms["appearance"] = {"value": None, "available": False,
                               "weight": None, "contribution": None}

    event_score = None
    if ball_centers:
        event_score = _ball_proximity(pred, actual, ball_centers)
    if event_score is not None:
        terms["event"] = {"value": event_score, "available": True,
                          "weight": None, "contribution": None}
    else:
        terms["event"] = {"value": None, "available": False,
                          "weight": None, "contribution": None}

    # --- pondération : renormalisation des présences réelles -----------------
    weights = dict(CORE_WEIGHTS)
    if appearance_available:
        weights["appearance"] = APPEARANCE_WEIGHT
    if event_score is not None:
        weights["event"] = OPTIONAL_EVENT_WEIGHT
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    score = 0.0
    for k, w in weights.items():
        val = terms[k].get("value")
        val = float(val) if val is not None else 0.0
        contrib = w * val
        terms[k]["weight"] = round_scale(w)
        terms[k]["contribution"] = round_scale(contrib)
        score += contrib

    return {
        "score": round_scale(score),
        "terms": terms,
        "weights": {k: round_scale(v) for k, v in weights.items()},
        "raw": {
            "gap_frames": int(gap),
            "predicted_center": [round_scale(pred[0], 3), round_scale(pred[1], 3)],
            "target_center": [round_scale(actual[0], 3), round_scale(actual[1], 3)],
            "spatial_distance_px": round_scale(dist, 3),
            "velocity_pred": [round_scale(v_pred[0], 3), round_scale(v_pred[1], 3)],
            "velocity_target": [round_scale(v_target[0], 3), round_scale(v_target[1], 3)],
            "velocity_delta": round_scale(vel_delta, 3),
            "size_delta_px": round_scale(size_delta, 3),
            "aspect_delta": round_scale(ar_delta, 4),
            "appearance_cosine": (round_scale(cos, 4) if cos is not None else None),
            "appearance_available": bool(appearance_available),
            "event_score": (round_scale(event_score, 4) if event_score is not None else None),
        },
        "_dist": dist,
        "_vel_delta": vel_delta,
        "_cos": cos,
    }


def _ball_proximity(pred, actual, ball_centers: dict[int, tuple[float, float]]) -> float | None:
    """Proximité à un ballon RÉEL (détecté). ``None`` si aucun ballon exploitable."""
    if not ball_centers:
        return None
    px = (pred[0] + actual[0]) / 2.0
    py = (pred[1] + actual[1]) / 2.0
    best = None
    for f in sorted(ball_centers):
        bx, by = ball_centers[f]
        d = math.hypot(px - bx, py - by)
        if best is None or d < best:
            best = d
    if best is None:
        return None
    return math.exp(-best / BALL_SCALE)


def _decide(scored: dict[str, Any]) -> tuple[str, str | None, float]:
    """Décision déterministe : ``(decision, rejection_reason, confidence)``."""
    raw = scored["raw"]
    gap = int(raw["gap_frames"])
    score = float(scored["score"])
    dist = float(scored["_dist"])
    vel_delta = float(scored["_vel_delta"])
    cos = scored["_cos"]

    if gap < MIN_GAP_FRAMES:
        return "REJECTED", "TEMPORAL_OVERLAP", 0.0
    if gap > MAX_GAP_FRAMES:
        return "REJECTED", "LONG_GAP", 0.0
    # Contradiction d'apparence forte (embeddings réels disponibles).
    if cos is not None and cos < APPEARANCE_MIN_COS:
        return "REJECTED", "CONTRADICTORY_APPEARANCE", 0.0
    # Contradiction spatiale : saut de position absolu rédhibitoire, ou saut
    # modéré aggravé par un mouvement incohérent.
    if dist > SPATIAL_REJECT_PX:
        return "REJECTED", "SPATIAL_VELOCITY_CONFLICT", 0.0
    if dist > SPATIAL_SOFT_PX and vel_delta > VELOCITY_REJECT:
        return "REJECTED", "SPATIAL_VELOCITY_CONFLICT", 0.0
    if score < ACCEPT_THRESHOLD:
        return "REJECTED", "BELOW_THRESHOLD", round_scale(score)
    return "ACCEPTED", None, round_scale(score)


# ---------------------------------------------------------------------------
# Analyse complète
# ---------------------------------------------------------------------------
def _candidate_pairs(frags: dict[int, Fragment]) -> list[tuple[int, int]]:
    ids = sorted(frags)
    pairs: list[tuple[int, int]] = []
    for a_id in ids:
        for b_id in ids:
            if a_id == b_id:
                continue
            a, b = frags[a_id], frags[b_id]
            if b.first_frame <= a.last_frame:
                continue  # pas de chevauchement : une suite doit être postérieure
            if b.first_frame - a.last_frame - 1 > MAX_GAP_FRAMES - 1:
                continue
            pairs.append((a_id, b_id))
    return pairs


def link_fragments(tracks: dict[Any, list], *,
                   embeddings: dict[int, Any] | None = None,
                   ball_detections: list[dict] | None = None,
                   width: float = 0.0, height: float = 0.0,
                   ) -> dict[str, Any]:
    """Analyse de continuité : liens évalués + lignées canoniques déterministes.

    Renvoie le rapport complet (schéma ``goalreel.track_continuity.v1``) sans le
    bloc héros (ajouté par :func:`hero_identity`).
    """
    frags = build_fragments(tracks, width=width, height=height)
    ball_centers = _ball_centers(ball_detections)

    # --- 1) évaluation de tous les liens candidats --------------------------
    evaluated: list[dict[str, Any]] = []
    for a_id, b_id in _candidate_pairs(frags):
        scored = score_link(frags[a_id], frags[b_id], embeddings=embeddings,
                            ball_centers=ball_centers)
        decision, reason, confidence = _decide(scored)
        evaluated.append({
            "source_track_id": int(a_id),
            "target_track_id": int(b_id),
            "gap_frames": int(scored["raw"]["gap_frames"]),
            "score": round_scale(scored["score"]),
            "confidence": round_scale(confidence),
            "decision": decision,
            "rejection_reason": reason,
            "evidence": scored["terms"],
            "raw": {k: v for k, v in scored["raw"].items()},
            "weights": scored["weights"],
        })

    # --- 2) affectation gloutonne déterministe ------------------------------
    accepted = [e for e in evaluated if e["decision"] == "ACCEPTED"]
    accepted.sort(key=lambda e: (-float(e["score"]),
                                 int(e["source_track_id"]),
                                 int(e["target_track_id"])))
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    linked: list[dict[str, Any]] = []
    for e in accepted:
        s, t = int(e["source_track_id"]), int(e["target_track_id"])
        if s in used_src or t in used_tgt:
            e["decision"] = "REJECTED"
            e["rejection_reason"] = "SUPERSEDED_BY_STRONGER_LINK"
            continue
        used_src.add(s)
        used_tgt.add(t)
        linked.append(e)

    # --- 3) construction des lignées ----------------------------------------
    succ = {int(e["source_track_id"]): int(e["target_track_id"]) for e in linked}
    pred = {int(e["target_track_id"]): int(e["source_track_id"]) for e in linked}
    link_by_src = {int(e["source_track_id"]): e for e in linked}

    identities: list[dict[str, Any]] = []
    visited: set[int] = set()
    for tid in sorted(frags):           # ordre déterministe
        if tid in visited:
            continue
        # remonter au début de la chaîne
        head = tid
        guard = 0
        while head in pred and guard < len(frags) + 1:
            head = pred[head]
            guard += 1
        chain: list[int] = []
        cur = head
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            visited.add(cur)
            chain.append(cur)
            cur = succ.get(cur)
        identities.append(_build_identity(chain, frags, link_by_src))

    report = {
        "schema": SCHEMA,
        "policy": POLICY,
        "config": {
            "core_weights": CORE_WEIGHTS,
            "appearance_weight": APPEARANCE_WEIGHT,
            "optional_event_weight": OPTIONAL_EVENT_WEIGHT,
            "accept_threshold": ACCEPT_THRESHOLD,
            "max_gap_frames": MAX_GAP_FRAMES,
            "appearance_min_cosine": APPEARANCE_MIN_COS,
            "spatial_reject_px": SPATIAL_REJECT_PX,
            "spatial_soft_px": SPATIAL_SOFT_PX,
            "velocity_reject": VELOCITY_REJECT,
            "scales": {"temporal": TEMPORAL_SCALE, "spatial": SPATIAL_SCALE,
                       "velocity": VELOCITY_SCALE, "geometry": GEOMETRY_SCALE,
                       "persistence": PERSISTENCE_SCALE, "ball": BALL_SCALE},
        },
        "raw_tracks": [_fragment_dict(frags[t]) for t in sorted(frags)],
        "links": evaluated,           # TOUS les liens évalués (audit complet)
        "accepted_links": [
            {k: e[k] for k in ("source_track_id", "target_track_id", "score",
                               "confidence", "gap_frames")}
            for e in linked
        ],
        "identities": identities,
        "appearance": {
            "available": bool(embeddings),
            "source": ("osnet" if embeddings else None),
            "tracks_with_embedding": len([t for t in (embeddings or {})]),
            "note": ("OSNet embeddings provided by the existing ReID pipeline; "
                     "appearance term renormalised when unavailable."),
            "fallback": (None if embeddings else FALLBACK_NO_APPEARANCE),
        },
        "sam2": _sam2_block(),
        "stats": {
            "raw_tracks": len(frags),
            "identity_count": len(identities),
            "links_evaluated": len(evaluated),
            "links_accepted": len(linked),
            "tracks_merged": int(sum(len(i["source_track_ids"]) - 1 for i in identities)),
        },
    }
    return report


def _sam2_block() -> dict[str, Any]:
    """État honnête de SAM2 comme preuve de continuité (jamais simulée).

    SAM2 n'est **pas** un oracle d'identité : il n'est utilisé que s'il est
    réellement disponible. Par défaut, il est marqué indisponible et la
    continuité se poursuit avec les autres preuves mesurées.
    """
    return {
        "available": False,
        "source": None,
        "fallback": FALLBACK_NO_SAM2,
        "note": ("SAM2 is supporting evidence only, never an identity oracle; "
                 "not run over the whole video for continuity."),
    }


def _fragment_dict(f: Fragment) -> dict[str, Any]:
    return {
        "track_id": int(f.track_id),
        "first_frame": int(f.first_frame),
        "last_frame": int(f.last_frame),
        "frames": int(f.length),
        "span": int(f.center_span()),
        "continuity": round_scale(f.continuity()),
        "mean_center": [round_scale(f.mean_center[0], 3), round_scale(f.mean_center[1], 3)],
        "mean_size": [round_scale(f.mean_size[0], 3), round_scale(f.mean_size[1], 3)],
        "confidence_mean": round_scale(f.conf_mean, 4),
        "person_sized": bool(f.person_sized),
    }


def _build_identity(chain: list[int], frags: dict[int, Fragment],
                    link_by_src: dict[int, dict[str, Any]],
                    canonical_track_id: int | None = None,
                    extra_fallback: str | None = None) -> dict[str, Any]:
    """Construit une identité canonique à partir d'une chaîne de fragments."""
    src_ids = [int(t) for t in chain]
    ranges = [{"start": int(frags[t].first_frame), "end": int(frags[t].last_frame),
               "track_id": int(t)} for t in chain]
    links = []
    confs = []
    for a, b in zip(chain, chain[1:]):
        e = link_by_src.get(int(a), {})
        links.append({
            "source_track_id": int(a),
            "target_track_id": int(b),
            "score": round_scale(e.get("score", 0.0)),
            "gap_frames": int(e.get("gap_frames", 0)),
            "evidence": e.get("evidence", {}),
        })
        confs.append(float(e.get("score", 0.0)))
    confidence = 1.0 if not confs else round_scale(sum(confs) / len(confs))
    canonical = int(canonical_track_id) if canonical_track_id is not None else int(src_ids[0])

    evidence = [f"fragments:{len(src_ids)}", f"range:{ranges[0]['start']}-{ranges[-1]['end']}"]
    for lk in links:
        evidence.append(f"link:{lk['source_track_id']}->{lk['target_track_id']}")

    fallback = extra_fallback
    if len(src_ids) == 1 and fallback is None:
        fallback = None

    return {
        "canonical_track_id": canonical,
        "source_track_ids": src_ids,
        "frame_ranges": ranges,
        "links": links,
        "confidence": confidence,
        "evidence": evidence,
        "fallback": fallback,
    }


# ---------------------------------------------------------------------------
# Héros
# ---------------------------------------------------------------------------
def identity_containing(report: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    """Identité canonique contenant ``track_id`` (ou ``None``)."""
    for ident in report.get("identities", []):
        if int(track_id) in {int(t) for t in ident.get("source_track_ids", [])}:
            return ident
    return None


def hero_identity(report: dict[str, Any], hero_track: int | None,
                  frags: dict[int, Fragment] | None = None,
                  tracks: dict[Any, list] | None = None) -> dict[str, Any]:
    """Promeut le héros brut en **identité canonique** (lignée).

    Si le héros est un fragment isolé, l'identité dégénère proprement :
    ``source_track_ids = [hero]`` et la confiance vaut ``1.0`` (aucune fusion
    nécessaire). Le contrat ``hero_track`` est préservé : ``canonical_track_id``
    vaut l'id brut du héros (donc ``hero_track == canonical hero identity``).

    Si aucune continuation crédible n'existe (preuve insuffisante), le champ
    ``fallback`` vaut ``UNRESOLVED_TRACK_FRAGMENT`` (fusion refusée, jamais
    fabriquée).
    """
    if hero_track is None:
        return {"canonical_track_id": None, "source_track_ids": [],
                "frame_ranges": [], "links": [], "confidence": 0.0,
                "evidence": [], "fallback": "NO_HERO_TRACK"}
    if frags is None:
        frags = build_fragments(tracks or {})
    ident = identity_containing(report, int(hero_track))
    if ident is None:
        f = frags.get(int(hero_track))
        if f is None:
            return {"canonical_track_id": int(hero_track), "source_track_ids": [int(hero_track)],
                    "frame_ranges": [], "links": [], "confidence": 0.0,
                    "evidence": ["hero_fragment_missing"], "fallback": "HERO_FRAGMENT_MISSING"}
        ident = _build_identity([int(hero_track)], frags, {},
                                canonical_track_id=int(hero_track))

    out = dict(ident)
    out["canonical_track_id"] = int(hero_track)
    out["method"] = "subject_continuity_v2.1"
    # Fallback renseigné : la lignée héros ne s'est-elle pas étendue alors qu'un
    # fragment de continuation proche a été REJETÉ (preuve insuffisante) ?
    if len(out["source_track_ids"]) <= 1:
        rejected_near = _rejected_near_continuation(report, hero_track)
        if rejected_near:
            out["fallback"] = FALLBACK_UNRESOLVED
            out["fallback_detail"] = rejected_near
    return out


def _rejected_near_continuation(report: dict[str, Any], hero_track: int) -> dict[str, Any] | None:
    """Premier lien rejeté partant de la fin du fragment héros (audit fallback)."""
    cands = [e for e in report.get("links", [])
             if int(e.get("source_track_id", -1)) == int(hero_track)
             and e.get("decision") == "REJECTED"]
    if not cands:
        return None
    cands.sort(key=lambda e: (int(e.get("gap_frames", 0)),
                              int(e.get("target_track_id", 0))))
    e = cands[0]
    return {"target_track_id": int(e["target_track_id"]),
            "reason": e.get("rejection_reason"), "score": e.get("score")}


# ---------------------------------------------------------------------------
# Résolution fragment actif (identité → fragment réel à une frame donnée)
# ---------------------------------------------------------------------------
def active_track_at(identity: dict[str, Any], frame: int) -> int | None:
    """Fragment réel actif d'une identité à ``frame`` (ou ``None``)."""
    if not identity:
        return None
    f = int(frame)
    for rng in identity.get("frame_ranges", []):
        if int(rng["start"]) <= f <= int(rng["end"]):
            return int(rng["track_id"])
    return None


def identity_frames(tracks: dict[Any, list], identity: dict[str, Any]) -> list[dict[str, Any]]:
    """Frames réelles concaténées des fragments d'une identité (triées).

    Chaque item porte ``track_id`` = fragment réel d'origine : la caméra reste
    couplée au **fragment actif** de l'identité canonique, jamais à un id inventé.
    """
    ids = {int(t) for t in identity.get("source_track_ids", [])}
    out: list[dict[str, Any]] = []
    for raw_id in sorted(tracks or {}, key=lambda k: int(k)):
        if int(raw_id) not in ids:
            continue
        for item in (tracks[raw_id] or []):
            if not isinstance(item, dict):
                continue
            f = _frame_of(item)
            b = _bbox_of(item)
            if f is None or b is None:
                continue
            out.append({"frame": int(f), "bbox": list(b),
                        "track_id": int(raw_id),
                        "confidence": float(item.get("confidence", 0.0) or 0.0)})
    out.sort(key=lambda it: (int(it["frame"]), int(it["track_id"])))
    return out


def _ball_centers(ball_detections: list[dict] | None) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for b in (ball_detections or []):
        if not isinstance(b, dict) or b.get("bbox") is None or b.get("frame") is None:
            continue
        bb = b["bbox"]
        try:
            out[int(b["frame"])] = ((float(bb[0]) + float(bb[2])) / 2.0,
                                    (float(bb[1]) + float(bb[3])) / 2.0)
        except (TypeError, ValueError, IndexError):
            continue
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Façade haut niveau
# ---------------------------------------------------------------------------
def analyze_continuity(tracks: dict[Any, list], *,
                       embeddings: dict[int, Any] | None = None,
                       ball_detections: list[dict] | None = None,
                       hero_track: int | None = None,
                       width: float = 0.0, height: float = 0.0) -> dict[str, Any]:
    """Analyse complète + identité héros. Renvoie le rapport sérialisable."""
    report = link_fragments(tracks, embeddings=embeddings,
                            ball_detections=ball_detections,
                            width=width, height=height)
    frags = build_fragments(tracks, width=width, height=height)
    report["hero_identity"] = hero_identity(report, hero_track, frags=frags,
                                            tracks=tracks)
    return report
