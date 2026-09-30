"""Tests de *cohérence caméra <-> preuve* (visual QA / tracked reframe).

Ces tests verrouillent les garanties issues du passage de QA visuelle :

  * la caméra 9:16 suit le sujet du MOMENT HÉROS lorsque sa trajectoire
    réelle existe (au lieu d'un choix arbitraire « plus longue trajectoire ») ;
  * le héros est départagé de façon *déterministe* via la persistance et
    l'amplitude réelles (la confiance ``min(0.99, disp/300)`` sature et crée de
    nombreuses égalités) ;
  * les centres suivis sont bornés pour que la fenêtre 9:16 reste DANS la source
    (aucune bordure noire, aucune coordonnée inventée) ;
  * l'absence de preuve conserve le fallback statique documenté.
"""
import pytest

from goalreel.cinematic.reframe import build_reframe_targets, clamp_targets
from goalreel.events.hero import score_hero, track_stats


def _event(event_id, track_id, displacement=371.62, confidence=0.99):
    return {
        "event_id": event_id, "kind": "player_motion",
        "start_frame": 0, "peak_frame": 1, "end_frame": 2,
        "confidence": confidence,
        "evidence": [f"track:{track_id}", f"displacement:{displacement:.2f}"],
        "status": "INFERRED",
    }


# --- sélection héros (départage déterministe) ------------------------------

def test_hero_breaks_confidence_tie_by_persistence():
    """Deux événements à 0.99 : le plus persistant doit gagner (pas arbitraire)."""
    tracks = {
        3: [{"frame": i, "bbox": [100 + i, 0, 120 + i, 10]} for i in range(5)],
        7: [{"frame": i, "bbox": [100 + i, 0, 120 + i, 10]} for i in range(40)],
    }
    events = [_event("MOTION_3", 3), _event("MOTION_7", 7)]
    hero = score_hero(events, track_stats=track_stats(tracks))
    assert hero["status"] == "OK"
    assert hero["track_id"] == 7
    assert hero["selection"]["persistence_frames"] == 40


def test_hero_tie_is_deterministic_and_stable():
    """Égalité complète => départage déterministe (id d'événement), stable entre appels."""
    tracks = {5: [{"frame": i, "bbox": [0, 0, 10, 10]} for i in range(10)],
              9: [{"frame": i, "bbox": [0, 0, 10, 10]} for i in range(10)]}
    events = [_event("MOTION_5", 5, displacement=100.0),
              _event("MOTION_9", 9, displacement=100.0)]
    a = score_hero(events, track_stats=track_stats(tracks))["track_id"]
    b = score_hero(events, track_stats=track_stats(tracks))["track_id"]
    # Résultat stable et non arbitraire : la clé documentée tranche (MOTION_9).
    assert a == b == 9
    # L'ordre d'entrée ne doit PAS changer le résultat (tri complet, pas de biais).
    c = score_hero(list(reversed(events)), track_stats=track_stats(tracks))["track_id"]
    assert c == a


def test_hero_prefers_higher_displacement_on_equal_persistence():
    tracks = {1: [{"frame": i, "bbox": [0, 0, 10, 10]} for i in range(10)]}
    events = [{"event_id": "A", "kind": "player_motion", "start_frame": 0,
               "peak_frame": 1, "end_frame": 2, "confidence": 0.5,
               "evidence": ["track:1", "displacement:50.00"], "status": "INFERRED"},
              {"event_id": "B", "kind": "player_motion", "start_frame": 0,
               "peak_frame": 1, "end_frame": 2, "confidence": 0.5,
               "evidence": ["track:1", "displacement:400.00"], "status": "INFERRED"}]
    hero = score_hero(events, track_stats=track_stats(tracks))
    assert hero["event"]["event_id"] == "B"


def test_hero_no_evidence_stays_unknown():
    assert score_hero([])["status"] == "UNKNOWN"


def test_track_stats_are_factual():
    tracks = {1: [{"frame": 0, "bbox": [0, 0, 10, 10]},
                  {"frame": 1, "bbox": [30, 0, 40, 10]}]}
    st = track_stats(tracks)
    assert st[1]["len"] == 2
    assert st[1]["displacement"] == pytest.approx(30.0)


# --- couplage caméra <-> héros ---------------------------------------------

def test_reframe_follows_hero_track_even_when_shorter():
    """La caméra suit le sujet héros même s'il n'est pas la trajectoire la plus longue."""
    tracks = {
        2: [{"frame": i, "bbox": [10, 0, 30, 10]} for i in range(300)],   # la plus longue
        7: [{"frame": i, "bbox": [500, 0, 520, 10]} for i in range(10)],  # le héros
    }
    targets, meta = build_reframe_targets(tracks, preferred_track=7)
    assert meta["source"] == "REAL_TRACKS"
    assert meta["mode"] == "FOLLOW_TRACKED_SUBJECT"
    assert meta["selection"] == "HERO_TRACK"
    assert meta["followed_track"] == 7
    assert targets[0] == pytest.approx(510.0)
    assert set(targets) == set(range(10))


def test_reframe_falls_back_to_longest_when_hero_unknown():
    tracks = {2: [{"frame": i, "bbox": [10, 0, 30, 10]} for i in range(300)],
              7: [{"frame": i, "bbox": [500, 0, 520, 10]} for i in range(10)]}
    targets, meta = build_reframe_targets(tracks, preferred_track=999)
    assert meta["selection"] == "LONGEST_TRACK"
    assert meta["followed_track"] == 2


def test_reframe_default_is_backward_compatible():
    """Sans héros fourni : comportement historique (trajectoire la plus longue)."""
    tracks = {3: [{"frame": 0, "bbox": [0, 0, 10, 10]}],
              7: [{"frame": 0, "bbox": [100, 0, 120, 10]},
                  {"frame": 1, "bbox": [130, 0, 150, 10]},
                  {"frame": 2, "bbox": [160, 0, 180, 10]}]}
    targets, meta = build_reframe_targets(tracks)
    assert meta["followed_track"] == 7
    assert targets[0] == pytest.approx(110.0)
    assert targets[2] == pytest.approx(170.0)
    assert set(targets) == {0, 1, 2}


def test_reframe_no_tracks_is_documented_static_fallback():
    targets, meta = build_reframe_targets({})
    assert targets == {}
    assert meta["source"] == "NO_TRACKS"
    assert meta["mode"] == "STATIC_FALLBACK"
    assert meta["selection"] == "NONE"
    assert meta["followed_track"] is None


# --- bornage (aucune bordure noire) ---------------------------------------

def test_clamp_keeps_window_inside_source():
    c = clamp_targets({0: 10.0, 1: 1000.0, 2: 512.0}, width=1024, crop_w=324)
    assert c[0] == pytest.approx(162.0)   # bord gauche <= crop_w/2
    assert c[1] == pytest.approx(862.0)   # bord droit  <= width-crop_w/2
    assert c[2] == pytest.approx(512.0)   # intact


def test_clamp_centers_when_crop_wider_than_source():
    c = clamp_targets({0: 10.0, 5: 90.0}, width=100, crop_w=324)
    assert c[0] == pytest.approx(50.0) and c[5] == pytest.approx(50.0)


def test_clamp_empty_is_empty():
    assert clamp_targets({}, width=1024, crop_w=324) == {}
