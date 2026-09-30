import cv2, numpy as np


def build_reframe_targets(tracks, preferred_track=None):
    """Construit les cibles caméra 9:16 à partir des trajectoires RÉELLES suivies.

    ``tracks`` : mapping ``track_id -> [ {frame, bbox, ...}, ... ]`` (issu du
    suivi ByteTrack sur de vraies détections YOLO).

    ``preferred_track`` : si fourni (ex. le sujet du *moment héros*), il devient
    le sujet suivi **dès lors qu'une trajectoire réelle existe pour cet id**.
    Cela garantit la cohérence « la caméra suit le sujet du moment héros » au
    lieu de choisir arbitrairement la trajectoire la plus longue. En son absence
    (ou si l'id est inconnu), on retombe sur la trajectoire la plus persistante.

    Renvoie ``(targets, meta)`` où ``targets`` est ``{frame_idx: cx}`` (centre
    horizontal réel, en pixels source) et ``meta`` décrit honnêtement la source.

    Politique : on **suit réellement** un joueur suivi. Aucune coordonnée n'est
    inventée : les trous temporels sont comblés par maintien de la dernière
    position réelle connue, et l'absence totale de trajectoire renvoie
    ``targets={}`` (=> fallback statique, jamais un faux suivi).
    """
    if not tracks:
        return {}, {"source": "NO_TRACKS", "mode": "STATIC_FALLBACK",
                    "followed_track": None, "target_frames": 0, "selection": "NONE"}

    # Sujet suivi : priorité au sujet du moment héros (preuve), sinon le plus
    # persistant. On ne suit un track préféré que s'il a une trajectoire réelle.
    if preferred_track is not None and tracks.get(preferred_track):
        primary_id = preferred_track
        selection = "HERO_TRACK"
    else:
        primary_id = max(tracks, key=lambda tid: len(tracks[tid]))
        selection = "LONGEST_TRACK"

    per_frame = {}
    for item in tracks[primary_id]:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        fr = item.get("frame")
        if bbox is None or fr is None:
            continue
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        per_frame[int(fr)] = cx
    if not per_frame:
        return {}, {"source": "NO_TRACKS", "mode": "STATIC_FALLBACK",
                    "followed_track": None, "target_frames": 0, "selection": "NONE"}
    return per_frame, {
        "source": "REAL_TRACKS",
        "mode": "FOLLOW_TRACKED_SUBJECT",
        "followed_track": int(primary_id),
        "followed_track_frames": len(tracks[primary_id]),
        "target_frames": len(per_frame),
        "selection": selection,
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
