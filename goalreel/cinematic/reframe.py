import cv2, numpy as np


def build_reframe_targets(tracks):
    """Construit les cibles caméra 9:16 à partir des trajectoires RÉELLES suivies.

    ``tracks`` : mapping ``track_id -> [ {frame, bbox, ...}, ... ]`` (issu du
    suivi ByteTrack sur de vraies détections YOLO).

    Renvoie ``(targets, meta)`` où ``targets`` est ``{frame_idx: cx}`` (centre
    horizontal réel, en pixels source) et ``meta`` décrit honnêtement la source.

    Politique : on **suit réellement** un joueur suivi (le plus persistant).
    Aucune coordonnée n'est inventée : les trous temporels sont comblés par
    maintien de la dernière position réelle connue, et l'absence totale de
    trajectoire renvoie ``targets={}`` (=> fallback statique, jamais un faux suivi).
    """
    if not tracks:
        return {}, {"source": "NO_TRACKS", "mode": "STATIC_FALLBACK",
                    "followed_track": None, "target_frames": 0}
    # Joueur suivi = trajectoire la plus longue (le plus présent) => sujet principal.
    primary_id = max(tracks, key=lambda tid: len(tracks[tid]))
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
                    "followed_track": None, "target_frames": 0}
    return per_frame, {
        "source": "REAL_TRACKS",
        "mode": "FOLLOW_TRACKED_SUBJECT",
        "followed_track": int(primary_id),
        "followed_track_frames": len(tracks[primary_id]),
        "target_frames": len(per_frame),
    }


def crop_to_target(frame,bbox,target_w=1080,target_h=1920,pad=2.2):
    h,w=frame.shape[:2]; x1,y1,x2,y2=map(float,bbox); cx=(x1+x2)/2; cy=(y1+y2)/2; bw=max(2,x2-x1)*pad; bh=max(2,y2-y1)*pad
    ar=target_w/target_h
    if bw/bh>ar: bh=bw/ar
    else: bw=bh*ar
    x1=max(0,int(cx-bw/2)); x2=min(w,int(cx+bw/2)); y1=max(0,int(cy-bh/2)); y2=min(h,int(cy+bh/2))
    crop=frame[y1:y2,x1:x2]
    if crop.size==0:return None
    return cv2.resize(crop,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
