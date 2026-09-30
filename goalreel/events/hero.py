"""Sélection du « moment héros » — contrainte par la PREUVE.

Politique ``NO_EVENT_WITHOUT_EVIDENCE`` : le héros est choisi uniquement à partir
d'événements réellement inférés (trajectoires suivies). Le classement combine :

  * la confiance de l'événement (déjà bornée) ;
  * un bonus pour un statut ``VERIFIED`` (rétro-compatible) ;
  * la **persistance réelle** de la trajectoire (``track_stats``) — un sujet
    présent plus longtemps est un meilleur héros qu'un sujet furtif ;
  * l'**amplitude** réelle du déplacement (preuve ``displacement:``).

Le départage est **déterministe** (persistance, amplitude, puis id d'événement)
afin d'éviter la sélection arbitraire provoquée par la saturation de la
confiance (``min(0.99, disp/300)``) qui crée de nombreuses égalités.
"""


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


def track_stats(tracks):
    """Statistiques *factuelles* par trajectoire : persistance + amplitude.

    ``tracks`` : ``{track_id: [ {frame,bbox}, ... ]}``. Aucune valeur inventée :
    la persistance est le nombre de frames réellement suivies et l'amplitude le
    déplacement réel du centre de la bbox entre la première et la dernière frame.
    """
    stats = {}
    for tid, frames in (tracks or {}).items():
        boxes = [f.get("bbox") if isinstance(f, dict) else f for f in frames]
        boxes = [b for b in boxes if b is not None]
        length = len(frames) if frames is not None else 0
        if not boxes:
            stats[tid] = {"len": length, "displacement": 0.0}
            continue
        xs = [(b[0] + b[2]) / 2 for b in boxes]
        ys = [(b[1] + b[3]) / 2 for b in boxes]
        disp = ((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2) ** 0.5
        stats[tid] = {"len": length, "displacement": float(disp)}
    return stats


def score_hero(events, track_stats=None):
    """Renvoie le moment héros + le ``track_id`` suivi (pour cohérence caméra)."""
    if not events:
        return {"status": "UNKNOWN", "reason": "NO_EVIDENCE"}

    ranked = []
    for e in events:
        ev_conf = float(e.get("confidence", 0))
        score = ev_conf
        if e.get("status") == "VERIFIED":
            score += 0.2
        tid = _track_id(e)
        persistence = 0.0
        if track_stats and tid is not None and tid in track_stats:
            persistence = float(track_stats[tid].get("len", 0))
        amplitude = 0.0
        if track_stats and tid is not None and tid in track_stats:
            amplitude = float(track_stats[tid].get("displacement", 0.0))
        if amplitude == 0.0:
            amplitude = _displacement(e)
        # Clé de tri enrichie : preuve, persistance, amplitude, id (déterministe).
        ranked.append((score, persistence, amplitude, str(e.get("event_id", "")), e))

    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best_score, best_pers, best_amp, _, best_event = ranked[0]
    hero_track = _track_id(best_event)
    return {
        "status": "OK",
        "event": best_event,
        "score": best_score,
        "track_id": hero_track,
        "method": "evidence_score",
        "ranked_by": ["evidence", "persistence", "amplitude", "event_id"],
        "candidates": len(events),
        "selection": {
            "persistence_frames": int(best_pers),
            "amplitude_px": round(float(best_amp), 2),
        },
    }
