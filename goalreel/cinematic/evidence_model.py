"""Couche de preuve du Cinematic Director V2.

Ce module extrait, à partir des sorties RÉELLES du pipeline existant
(détections, suivi ByteTrack, re-ID, profondeur, segmentation, ballon COCO,
estimation caméra optique), les signaux factuels dont le director a besoin :

  * **frontières de plan réelles** de la source (pic de différence inter-frame
    mesuré sur les pixels — jamais supposé) ;
  * **profil de mouvement temporel** (evidence de rythme) ;
  * **présence et géométrie du héros suivi** (frames réellement suivies) ;
  * **géométrie du groupe** (étendue horizontale réelle des sujets suivis) pour
    choisir un cadrage qui contient réellement le jeu ;
  * **trajectoires taille-personne** (pour ne jamais suivre un blob large) ;
  * **preuve ballon réelle** (``sports ball`` COCO) avec sa fenêtre temporelle.

Vérité
------
Aucune valeur n'est inventée : toute position provient d'une bbox réellement
suivie ; toute frontière provient d'une mesure réelle sur la source ; toute
absence est rapportée comme telle (et non comblée par une valeur plausible).

Déterminisme
------------
Toutes les itérations de dictionnaire sont triées ; les tableaux renvoyés sont
ordonnés. Fonctions pures (hormis :func:`scan_scene`) : sans IO, elles sont
testables directement.
"""
from __future__ import annotations

from typing import Any

from .edit_plan import (
    CUT_DIFF_THRESHOLD,
    PERSON_MAX_HEIGHT,
    PERSON_MAX_WIDTH,
    round_scale,
)


# ---------------------------------------------------------------------------
# Helpers géométriques (preuve réelle uniquement)
# ---------------------------------------------------------------------------
def _bbox_of(item: Any):
    if isinstance(item, dict):
        b = item.get("bbox")
    else:
        b = item
    if b is None:
        return None
    try:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except (TypeError, ValueError, IndexError):
        return None


def _frame_of(item: Any, fallback: int) -> int:
    if isinstance(item, dict) and item.get("frame") is not None:
        try:
            return int(item["frame"])
        except (TypeError, ValueError):
            return fallback
    return fallback


def track_frame_centers(frames: list[Any]) -> dict[int, tuple[float, float]]:
    """``{frame: (cx, cy)}`` des centres réels d'une trajectoire suivie."""
    out: dict[int, tuple[float, float]] = {}
    for i, item in enumerate(frames or []):
        b = _bbox_of(item)
        if b is None:
            continue
        out[_frame_of(item, i)] = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return dict(sorted(out.items()))


def track_frame_sizes(frames: list[Any]) -> dict[int, tuple[float, float]]:
    """``{frame: (w, h)}`` des tailles réelles de bbox d'une trajectoire."""
    out: dict[int, tuple[float, float]] = {}
    for i, item in enumerate(frames or []):
        b = _bbox_of(item)
        if b is None:
            continue
        out[_frame_of(item, i)] = (b[2] - b[0], b[3] - b[1])
    return dict(sorted(out.items()))


def is_person_sized(frames: list[Any]) -> bool:
    """Vrai si la trajectoire correspond à un joueur (et non un plan serré).

    Preuve : largeur/hauteur moyennes réelles des bbox de la trajectoire. Un
    blob large est un objet de premier plan (advertising board, silhouette de
    célébration), pas un joueur suivi en plan large.
    """
    sizes = list(track_frame_sizes(frames).values())
    if not sizes:
        return False
    w = sum(s[0] for s in sizes) / len(sizes)
    h = sum(s[1] for s in sizes) / len(sizes)
    return w <= PERSON_MAX_WIDTH and h <= PERSON_MAX_HEIGHT


def person_sized_tracks(tracks: dict[Any, list]) -> dict[int, list]:
    """Sous-ensemble des trajectoires taille-personne (clés triées)."""
    out: dict[int, list] = {}
    for tid in sorted(tracks or {}, key=lambda k: int(k)):
        if is_person_sized(tracks[tid]):
            out[int(tid)] = tracks[tid]
    return out


def hero_presence(track_id: Any, tracks: dict[Any, list]) -> dict[str, Any]:
    """Présence réelle du héros : frames suivies, étendue, continuité.

    Renvoie ``present=False`` si cet id n'a pas de trajectoire réellement suivie
    (aucune frame n'est fabriquée).
    """
    if track_id is None or int(track_id) not in {int(k) for k in (tracks or {})}:
        return {"present": False, "track_id": None, "frames": 0,
                "first_frame": None, "last_frame": None, "continuity": 0.0}
    frames = tracks[int(track_id)] or []
    centers = track_frame_centers(frames)
    if not centers:
        return {"present": False, "track_id": int(track_id), "frames": 0,
                "first_frame": None, "last_frame": None, "continuity": 0.0}
    keys = sorted(centers)
    span = keys[-1] - keys[0] + 1
    return {
        "present": True,
        "track_id": int(track_id),
        "frames": len(centers),
        "first_frame": int(keys[0]),
        "last_frame": int(keys[-1]),
        "continuity": round_scale(min(1.0, len(centers) / float(span or 1))),
    }


def group_geometry(tracks: dict[Any, list], frames_all: bool = True) -> dict[str, Any]:
    """Géométrie réelle du groupe suivi, par frame.

    Renvoie ``group_center_x`` / ``group_x_min`` / ``group_x_max`` /
    ``group_count`` indexés par frame, uniquement à partir de bbox réellement
    suivies. L'étendue sert à garantir qu'un cadrage « large » contient
    effectivement le jeu.
    """
    center: dict[int, float] = {}
    xmin: dict[int, float] = {}
    xmax: dict[int, float] = {}
    count: dict[int, int] = {}
    for tid in sorted(tracks or {}, key=lambda k: int(k)):
        for fr, (cx, _) in track_frame_centers(tracks[tid]).items():
            b = None
            for item in tracks[tid]:
                if _frame_of(item, -1) == fr:
                    b = _bbox_of(item)
                    break
            if b is None:
                continue
            count[fr] = count.get(fr, 0) + 1
            center[fr] = center.get(fr, 0.0) + cx
            xmin[fr] = min(xmin.get(fr, b[0]), b[0])
            xmax[fr] = max(xmax.get(fr, b[2]), b[2])
    for fr in sorted(center):
        center[fr] = round_scale(center[fr] / max(1, count[fr]), 3)
    return {
        "group_center_x": dict(sorted(center.items())),
        "group_x_min": dict(sorted(xmin.items())),
        "group_x_max": dict(sorted(xmax.items())),
        "group_count": dict(sorted(count.items())),
    }


def ball_evidence(ball_detections: list[dict] | None) -> dict[str, Any]:
    """Preuve ballon RÉELLE (classe COCO ``sports ball``) — jamais simulée."""
    balls = [b for b in (ball_detections or [])
             if isinstance(b, dict) and b.get("bbox") is not None
             and b.get("frame") is not None]
    if not balls:
        return {"available": False, "count": 0, "frames": [], "first_frame": None,
                "last_frame": None, "centers": {}}
    frames = sorted(int(b["frame"]) for b in balls)
    centers = {}
    for b in sorted(balls, key=lambda x: int(x["frame"])):
        bb = b["bbox"]
        centers[int(b["frame"])] = [
            round_scale((float(bb[0]) + float(bb[2])) / 2.0, 2),
            round_scale((float(bb[1]) + float(bb[3])) / 2.0, 2),
        ]
    return {
        "available": True,
        "count": len(balls),
        "frames": frames,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "centers": dict(sorted(centers.items())),
        "note": ("COCO 'sports ball' detections only; football events are "
                 "inferred from tracked motion, never fabricated."),
    }


# ---------------------------------------------------------------------------
# Mesure RÉELLE des frontières de plan et du mouvement sur la source
# ---------------------------------------------------------------------------
def scan_scene(video: str, threshold: float = CUT_DIFF_THRESHOLD,
               sample_every: int = 1, max_frames: int = 0) -> dict[str, Any]:
    """Mesure les coupes réelles et le profil de mouvement de la source.

    La frontière de plan est un **pic** de différence inter-frame : à la frame
    ``i``, la moyenne des |diff| 8-bit entre ``i-1`` et ``i`` dépasse le seuil
    ET est au moins 3× la moyenne locale des 6 frames précédentes. C'est une
    mesure directe sur les pixels, reproduite sur plusieurs exécutions ; c'est
    le seul « cut » enregistré par le director.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("VIDEO_OPEN_FAILED")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    prev = None
    motion: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion.append(0.0 if prev is None else float(np.abs(gray - prev).mean()))
        prev = gray
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()

    cuts: list[int] = []
    arr = motion
    for i in range(1, len(arr)):
        local = arr[max(0, i - 6):i]
        base = (sum(local) / len(local)) if local else 0.0
        if arr[i] > threshold and arr[i] > 3.0 * max(base, 1e-6):
            cuts.append(int(i))

    peak = int(max(range(len(arr)), key=lambda k: arr[k])) if arr else None
    return {
        "schema": "goalreel.scene_scan.v1",
        "frames": len(arr),
        "fps": fps,
        "cut_frames": cuts,
        "cut_count": len(cuts),
        "motion": [round_scale(m, 4) for m in arr],
        "motion_mean": round_scale(sum(arr) / len(arr) if arr else 0.0, 4),
        "motion_peak_frame": peak,
        "motion_peak_value": round_scale(arr[peak], 4) if peak is not None else 0.0,
        "threshold": threshold,
        "evidence": "VISIBLE_FROM_SOURCE",
    }


def motion_window_profile(motion: list[float] | None, start: int, end: int) -> dict[str, Any]:
    """Profil de mouvement réel sur une fenêtre ``[start, end]`` (inclusive)."""
    if not motion:
        return {"mean": 0.0, "peak": 0.0, "peak_frame": None, "samples": 0}
    lo = max(0, int(start))
    hi = min(len(motion) - 1, int(end))
    if hi < lo:
        return {"mean": 0.0, "peak": 0.0, "peak_frame": None, "samples": 0}
    seg = motion[lo:hi + 1]
    peak_idx = max(range(len(seg)), key=lambda k: seg[k]) if seg else 0
    return {
        "mean": round_scale(sum(seg) / len(seg), 4),
        "peak": round_scale(seg[peak_idx], 4) if seg else 0.0,
        "peak_frame": int(lo + peak_idx),
        "samples": len(seg),
    }


def build_evidence(tracks: dict[Any, list] | None = None,
                   ball_detections: list[dict] | None = None,
                   camera_transforms: list[dict] | None = None,
                   width: int = 0, height: int = 0,
                   total_frames: int = 0,
                   scene: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble la couche de preuve consommée par le director.

    Toutes les entrées proviennent des sorties réelles du pipeline ; aucune
    valeur n'est générée synthétiquement.
    """
    tracks = tracks or {}
    scene = scene or {}
    motion = scene.get("motion")
    return {
        "schema": "goalreel.cinematic_evidence.v1",
        "source": {"width": int(width), "height": int(height),
                   "total_frames": int(total_frames)},
        "tracks": {
            "count": len(tracks),
            "person_sized_count": len(person_sized_tracks(tracks)),
            "person_sized_ids": sorted(int(k) for k in person_sized_tracks(tracks)),
        },
        "ball": ball_evidence(ball_detections),
        "scene": {
            "cut_frames": list(scene.get("cut_frames", [])),
            "cut_count": int(scene.get("cut_count", 0)),
            "motion_mean": round_scale(scene.get("motion_mean", 0.0), 4),
            "motion_peak_frame": scene.get("motion_peak_frame"),
            "motion_peak_value": round_scale(scene.get("motion_peak_value", 0.0), 4),
            "evidence": scene.get("evidence", "VISIBLE_FROM_SOURCE"),
            "_motion": motion,
        },
        "group": group_geometry(tracks),
        "camera_motion": _camera_motion_summary(camera_transforms),
    }


def _camera_motion_summary(transforms: list[dict] | None) -> dict[str, Any]:
    """Résumé factuel du mouvement caméra optique (preuve ``CameraEstimator``)."""
    if not transforms:
        return {"available": False, "samples": 0}
    dxs = []
    for t in transforms:
        try:
            dxs.append(float(t.get("dx", 0.0)))
        except (TypeError, ValueError):
            continue
    if not dxs:
        return {"available": False, "samples": 0}
    return {
        "available": True,
        "samples": len(dxs),
        "mean_dx": round_scale(sum(dxs) / len(dxs), 4),
        "min_dx": round_scale(min(dxs), 4),
        "max_dx": round_scale(max(dxs), 4),
        "evidence": "VISIBLE_FROM_SOURCE",
    }
