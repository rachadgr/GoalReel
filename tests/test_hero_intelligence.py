"""Tests de régression — intelligence héros V2 (multi-preuves) & timing caméra.

Couverture ciblée exigée :

  * classement héros **déterministe** (indépendant de l'ordre d'entrée) ;
  * **pondération par preuve** explicable (breakdown, somme des poids = 1) ;
  * **proximité temporelle** à l'événement/pic d'action ;
  * **proximité au ballon** quand une preuve ballon RÉELLE existe ;
  * **fallback sans ballon** (aucune preuve inventée) ;
  * **persistance** de trajectoire ;
  * départage **stable** en cas d'égalité totale ;
  * contrat **hero_track == followed_track** ;
  * **fallback documenté** quand la trajectoire du héros est indisponible ;
  * **continuité** des cibles cinématiques (phases, pas de saut) ;
  * **bornage** (aucune bordure noire).

Toutes les données sont synthétiques et déterministes : aucun modèle requis.
"""
import pytest

from goalreel.cinematic.reframe import build_reframe_targets, clamp_targets
from goalreel.events.hero import (
    BASE_WEIGHTS,
    _sat,
    rank_candidates,
    score_hero,
    track_stats,
)


# --------------------------------------------------------------------------- #
# Helpers synthétiques
# --------------------------------------------------------------------------- #
def _track(tid, n, cx0, y=520, start=0, step=1):
    """Trajectoire réelle déterministe : centre balayé de ``step`` px/frame."""
    return [
        {"frame": start + i, "bbox": [cx0 + step * i - 10, y - 20,
                                      cx0 + step * i + 10, y + 20]}
        for i in range(n)
    ]


def _event(event_id, tid, peak, start=None, end=None, displacement=400.0,
           confidence=0.99):
    start = 0 if start is None else start
    end = (start + 2) if end is None else end
    return {
        "event_id": event_id, "kind": "player_motion",
        "start_frame": start, "peak_frame": peak, "end_frame": end,
        "confidence": confidence,
        "evidence": [f"track:{tid}", f"displacement:{displacement:.2f}"],
        "status": "INFERRED",
    }


# --------------------------------------------------------------------------- #
# Déterminisme & stabilité
# --------------------------------------------------------------------------- #
def test_hero_is_deterministic_and_order_independent():
    tracks = {1: _track(1, 40, 100), 2: _track(2, 40, 500), 3: _track(3, 80, 300)}
    events = [_event("MOTION_1", 1, 20), _event("MOTION_2", 2, 20),
              _event("MOTION_3", 3, 20)]
    st = track_stats(tracks)
    a = score_hero(events, track_stats=st, tracks=tracks, width=1024, height=576)
    b = score_hero(list(reversed(events)), track_stats=st, tracks=tracks,
                   width=1024, height=576)
    assert a["track_id"] == b["track_id"]
    assert a["status"] == "OK" and a["method"] == "multi_evidence_v2"


def test_hero_stable_tie_break_on_complete_equality():
    """Égalité parfaite => départage total & stable (persistance, amplitude, id)."""
    tracks = {5: _track(5, 20, 200), 9: _track(9, 20, 200)}
    events = [_event("MOTION_5", 5, 10, displacement=100.0),
              _event("MOTION_9", 9, 10, displacement=100.0)]
    st = track_stats(tracks)
    r1 = score_hero(events, track_stats=st, tracks=tracks, width=1024, height=576)
    r2 = score_hero(list(reversed(events)), track_stats=st, tracks=tracks,
                    width=1024, height=576)
    # Départage documenté : event_id décroissant (MOTION_9).
    assert r1["track_id"] == r2["track_id"] == 9


def test_rank_candidates_scores_are_explainable():
    tracks = {1: _track(1, 50, 100)}
    events = [_event("MOTION_1", 1, 25)]
    ranked, meta = rank_candidates(events, track_stats=track_stats(tracks),
                                   tracks=tracks, width=1024, height=576)
    cand = ranked[0]
    # Chaque contribution = poids * valeur, et la somme = score.
    total = 0.0
    for k, b in cand["breakdown"].items():
        assert b["contribution"] == pytest.approx(b["weight"] * b["value"], abs=1e-5)
        total += b["contribution"]
    assert total == pytest.approx(cand["score"], abs=1e-5)
    # Sans preuve ballon, les poids de base somment à 1 et AUCUN poids ballon.
    assert "ball_proximity" not in meta["weights"]
    assert sum(meta["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert meta["ball_evidence"] is False


# --------------------------------------------------------------------------- #
# Persistance & amplitude (preuve de suivi)
# --------------------------------------------------------------------------- #
def test_persistence_prefers_longer_real_track():
    tracks = {3: _track(3, 5, 100), 7: _track(7, 40, 100)}
    events = [_event("MOTION_3", 3, 2), _event("MOTION_7", 7, 20)]
    hero = score_hero(events, track_stats=track_stats(tracks), tracks=tracks,
                      width=1024, height=576)
    assert hero["track_id"] == 7
    assert hero["selection"]["persistence_frames"] == 40


def test_amplitude_saturation_is_monotonic_not_clamped():
    """La saturation d'amplitude croît strictement (plus de plateau à 0.99)."""
    vals = [_sat(x, 300.0) for x in (50, 150, 300, 600, 1200, 3000)]
    assert all(0.0 <= v < 1.0 for v in vals)
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_amplitude_breaks_tie_when_persistence_equal():
    tracks = {1: _track(1, 10, 100)}
    events = [_event("A", 1, 5, displacement=50.0),
              _event("B", 1, 5, displacement=400.0)]
    hero = score_hero(events, track_stats=track_stats(tracks), tracks=tracks,
                      width=1024, height=576)
    assert hero["event"]["event_id"] == "B"


# --------------------------------------------------------------------------- #
# Proximité temporelle à l'événement / au pic d'action
# --------------------------------------------------------------------------- #
def test_temporal_proximity_prefers_event_near_focus_peak():
    tracks = {1: _track(1, 60, 100), 2: _track(2, 60, 700)}
    # Pic d'action réel au frame 100 : l'événement proche (peak 100) doit gagner.
    events = [_event("NEAR", 1, 100, displacement=200.0),
              _event("FAR", 2, 500, displacement=200.0)]
    hero = score_hero(events, track_stats=track_stats(tracks), tracks=tracks,
                      width=1024, height=576, focus_peak=100)
    assert hero["event"]["event_id"] == "NEAR"
    assert hero["focus_peak_frame"] == 100


# --------------------------------------------------------------------------- #
# Proximité au ballon (preuve RÉELLE uniquement)
# --------------------------------------------------------------------------- #
def _ball_near_track_mid(track_items, tid):
    mid = track_items[len(track_items) // 2]
    b = mid["bbox"]
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    return [{"frame": mid["frame"], "bbox": [cx - 8, cy - 8, cx + 8, cy + 8],
             "confidence": 0.6}]


def test_ball_proximity_enabled_only_with_real_ball_evidence():
    tracks = {1: _track(1, 40, 500), 2: _track(2, 40, 60)}
    st = track_stats(tracks)
    bell = _ball_near_track_mid(tracks[1], 1)
    events = [_event("MOTION_1", 1, 20, displacement=200.0),
              _event("MOTION_2", 2, 20, displacement=200.0)]

    # Avec preuve ballon RÉELLE : le sujet proche du ballon gagne.
    with_ball = score_hero(events, track_stats=st, tracks=tracks,
                           ball_detections=bell, width=1024, height=576)
    assert with_ball["ball_evidence"] is True
    assert "ball_proximity" in with_ball["weights"]
    assert with_ball["track_id"] == 1
    assert with_ball["selection"]["ball_min_dist_px"] is not None
    assert with_ball["selection"]["ball_proximity_score"] > 0.9
    # Poids renormalisés : la somme reste 1.
    assert sum(with_ball["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_no_ball_fallback_uses_only_track_evidence():
    tracks = {1: _track(1, 40, 500), 2: _track(2, 40, 60)}
    st = track_stats(tracks)
    events = [_event("MOTION_1", 1, 20, displacement=200.0),
              _event("MOTION_2", 2, 20, displacement=200.0)]
    hero = score_hero(events, track_stats=st, tracks=tracks,
                      ball_detections=[], width=1024, height=576)
    # Aucune preuve ballon => poids ballon absent, ball_evidence False.
    assert hero["ball_evidence"] is False
    assert "ball_proximity" not in hero["weights"]
    assert hero["selection"]["ball_min_dist_px"] is None
    assert hero["selection"]["ball_proximity_score"] == 0.0
    assert sum(hero["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_ball_weight_matches_documented_base_weights():
    # Sanity : les poids de base sont exactement ceux documentés.
    assert set(BASE_WEIGHTS) == {
        "amplitude", "persistence", "continuity", "event_proximity",
        "spatial_relevance", "event_confidence",
    }
    assert sum(BASE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Contrat événement → héros → caméra
# --------------------------------------------------------------------------- #
def test_hero_track_equals_followed_track():
    tracks = {2: _track(2, 300, 100), 7: _track(7, 10, 900)}
    events = [_event("MOTION_2", 2, 150), _event("MOTION_7", 7, 5)]
    hero = score_hero(events, track_stats=track_stats(tracks), tracks=tracks,
                      width=1024, height=576)
    targets, meta = build_reframe_targets(
        tracks, preferred_track=hero["track_id"], hero_event=hero["event"],
        width=1024)
    assert hero["track_id"] == meta["followed_track"]
    # Contrat explicite par nom de champ (verifiable littéralement) :
    # hero_moment.hero_track == camera_reframe.followed_track
    assert hero["hero_track"] == meta["followed_track"]
    assert hero["hero_track"] == hero["track_id"]
    assert meta["selection"] == "HERO_TRACK"
    assert meta["reason"] == "HERO_TRACK_AVAILABLE"
    assert meta["target_frames"] == len(targets) > 0


def test_missing_preferred_track_documented_fallback():
    tracks = {2: _track(2, 100, 100), 7: _track(7, 10, 900)}
    _, meta = build_reframe_targets(tracks, preferred_track=9999, width=1024)
    assert meta["selection"] == "LONGEST_TRACK"
    assert meta["reason"] == "PREFERRED_TRACK_UNAVAILABLE"
    assert meta["followed_track"] == 2


def test_fallback_is_deterministic_on_equal_length_tracks():
    """Deux trajectoires de même longueur => sélection stable (id max)."""
    tracks = {3: _track(3, 20, 100), 8: _track(8, 20, 700)}
    _, meta = build_reframe_targets(tracks, preferred_track=None, width=1024)
    assert meta["selection"] == "LONGEST_TRACK"
    assert meta["followed_track"] == 8  # départage stable par id décroissant


# --------------------------------------------------------------------------- #
# Timing cinématique (phases) & continuité
# --------------------------------------------------------------------------- #
def test_cinematic_phases_are_ordered_and_continuous():
    items = _track(1, 200, 100, step=1)
    event = _event("MOTION_1", 1, 100, start=80, end=120)
    targets, meta = build_reframe_targets({1: items}, preferred_track=1,
                                          hero_event=event, width=1024)
    phases = meta["phases"]
    # Toutes les phases attendues sont présentes.
    assert {"anticipation", "hero_moment", "reaction", "celebration"} <= set(phases.values())
    ant = [f for f, p in phases.items() if p == "anticipation"]
    hero = [f for f, p in phases.items() if p == "hero_moment"]
    react = [f for f, p in phases.items() if p == "reaction"]
    celeb = [f for f, p in phases.items() if p == "celebration"]
    # Ordre strict des phases.
    assert max(ant) < min(hero) <= max(hero) < min(react) <= max(react) < min(celeb)
    # Continuité : aucun saut brusque de cible (trajectoire réelle step=1).
    assert meta["max_step_px"] <= 2.0


def test_cinematic_targets_are_always_real_centers_no_invention():
    """Chaque cible provient d'un centre réel ou de son maintien — jamais inventée."""
    items = _track(1, 60, 100, step=2, start=0)
    event = _event("MOTION_1", 1, 30, start=20, end=40)
    targets, meta = build_reframe_targets({1: items}, preferred_track=1,
                                          hero_event=event, width=1024)
    real_centers = set()
    for it in items:
        b = it["bbox"]
        real_centers.add((b[0] + b[2]) / 2.0)
    for f, cx in targets.items():
        assert cx in real_centers, f"target at frame {f} is not a real tracked center"


def test_cinematic_continuity_no_abrupt_target_change():
    # Longue trajectoire : la phase celebration ne doit pas créer de saut.
    items = _track(1, 300, 200, step=1)
    event = _event("MOTION_1", 1, 150, start=130, end=170)
    _, meta = build_reframe_targets({1: items}, preferred_track=1,
                                    hero_event=event, width=1024)
    # Le pas max reste celui de la trajectoire réelle (balayage de 1 px/frame).
    assert meta["max_step_px"] <= 1.0 + 1e-9


def test_no_event_keeps_plain_tracking_phases():
    items = _track(1, 50, 100, step=1)
    _, meta = build_reframe_targets({1: items}, preferred_track=1, width=1024)
    assert set(meta["phases"].values()) == {"track"}
    assert meta["event_window"] is None


# --------------------------------------------------------------------------- #
# Bornage (aucune bordure noire)
# --------------------------------------------------------------------------- #
def test_clamp_keeps_cinematic_targets_inside_source():
    items = _track(1, 120, 950, step=1)  # sujet proche du bord droit
    event = _event("MOTION_1", 1, 60, start=40, end=80)
    targets, _ = build_reframe_targets({1: items}, preferred_track=1,
                                       hero_event=event, width=1024)
    crop_w = round(576 * 9 / 16)
    clamped = clamp_targets(targets, width=1024, crop_w=crop_w)
    lo, hi = crop_w / 2.0, 1024 - crop_w / 2.0
    assert all(lo - 1e-6 <= v <= hi + 1e-6 for v in clamped.values())


def test_clamp_empty_is_empty():
    assert clamp_targets({}, width=1024, crop_w=324) == {}
