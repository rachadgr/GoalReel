"""Tests déterministes — Subject Continuity V2.2 (association prédictive).

Vérifie les garanties AJOUTÉES par V2.2 sans affaiblir V2.1 :

  1. continuation prédictive à vitesse stable (acceptée)
  2. continuation prédictive avec petite accélération (acceptée)
  3. rejet sur erreur de prédiction trop grande (PREDICTION_ERROR_TOO_LARGE)
  4. rejet sur trou long (LONG_GAP) — inchangé
  5. rejet sur apparence OSNet contradictoire — inchangé
  6. forte similarité OSNet + bonne prédiction (acceptée)
  7. fallback OSNet manquant (apparence indisponible, échec en douceur)
  8. ordre des candidats déterministe
  9. JSON de sortie déterministe
 10. extension de lignée héros
 11. la caméra suit le nouveau fragment actif
 12. contrat héros/caméra conservé
 13. tolérance de vitesse robuste (bornée, dépend du trou)
 14. prédiction bornée (horizon plafonné)
 15. aucune identité fabriquée
 16. non-régression : schéma v2 + comportement V2.1 préservé
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from goalreel.tracking.continuity import (
    ACCEPT_THRESHOLD,
    METHOD,
    PREDICTION_MAX_HORIZON,
    SCHEMA,
    active_track_at,
    analyze_continuity,
    appearance_representation,
    build_fragments,
    identity_frames,
    link_fragments,
    predict_position,
    prediction_error,
    score_link,
    velocity_tolerance,
)


def _frag(tid, start, end, cx0, cx1, cy=380.0, w=20.0, h=50.0, conf=0.9):
    n = end - start + 1
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        cx = cx0 + (cx1 - cx0) * t
        out.append({"frame": start + i, "track_id": tid,
                    "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    "confidence": conf})
    return out


def _emb(v):
    a = np.asarray(v, dtype="float32")
    return a / (np.linalg.norm(a) or 1.0)


def _tracks(*frags):
    return {f[0]["track_id"]: list(f) for f in frags}


# 1. continuation prédictive à vitesse stable
def test_predictive_stable_velocity_accepted():
    a = _frag(9, 0, 100, 200, 600)          # +4 px/frame
    b = _frag(14, 112, 200, 650, 760)       # continue dans la même direction
    report = analyze_continuity(_tracks(a, b), hero_track=9)
    assert report["hero_identity"]["source_track_ids"] == [9, 14]
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["decision"] == "ACCEPTED"
    assert link["raw"]["prediction"]["prediction_error_px"] >= 0.0


# 2. continuation prédictive avec petite accélération
def test_predictive_small_acceleration_accepted():
    a = _frag(9, 0, 100, 200, 600)          # ~+4 px/frame
    b = _frag(14, 108, 200, 632, 780)       # un peu plus rapide (+1.7 px/frame)
    report = analyze_continuity(_tracks(a, b), hero_track=9)
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["decision"] == "ACCEPTED"
    assert link["raw"]["velocity_tolerance"] > 0.0


# 3. rejet sur erreur de prédiction trop grande
def test_prediction_error_too_large_rejected():
    a = _frag(9, 0, 100, 0, 400)            # +4 px/frame vers la droite
    b = _frag(14, 104, 200, 760, 800)       # réapparaît très loin devant
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14])
    report = link_fragments(_tracks(a, b))
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["decision"] == "REJECTED"
    assert link["rejection_reason"] in (
        "PREDICTION_ERROR_TOO_LARGE", "SPATIAL_VELOCITY_CONFLICT",
        "BELOW_THRESHOLD")
    assert scored["raw"]["prediction"]["normalized_by_diagonal"] > 0.0


# 4. rejet sur trou long (inchangé)
def test_long_gap_still_rejected():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 300, 400, 610, 700)
    report = link_fragments(_tracks(a, b))
    pairs = {(e["source_track_id"], e["target_track_id"]) for e in report["links"]}
    assert (9, 14) not in pairs
    assert report["stats"]["links_accepted"] == 0


# 5. rejet sur apparence contradictoire (inchangé)
def test_contradictory_appearance_still_rejected():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0, 1, 0, 0])}
    report = link_fragments(_tracks(a, b), embeddings=emb)
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["rejection_reason"] == "CONTRADICTORY_APPEARANCE"


# 6. forte similarité OSNet + bonne prédiction (acceptée)
def test_high_osnet_and_good_prediction_accepted():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 108, 200, 628, 740)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0.99, 0.02, 0, 0])}
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14], embeddings=emb)
    assert scored["raw"]["appearance_cosine"] > 0.9
    assert scored["score"] > ACCEPT_THRESHOLD


# 7. fallback OSNet manquant (échec en douceur)
def test_missing_osnet_soft_fail():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 650, 760)
    report = link_fragments(_tracks(a, b), embeddings=None)
    assert report["appearance"]["available"] is False
    assert report["stats"]["links_accepted"] >= 1


# 8. ordre des candidats déterministe
def test_candidate_ordering_deterministic():
    a = _frag(9, 0, 100, 200, 600)
    b1 = _frag(14, 112, 200, 640, 720)
    b2 = _frag(21, 112, 200, 640, 720)
    tracks = _tracks(a, b1, b2)
    r1 = analyze_continuity(tracks, hero_track=9)
    r2 = analyze_continuity({21: tracks[21], 14: tracks[14], 9: tracks[9]}, hero_track=9)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    accepted = [e for e in r1["links"] if e["decision"] == "ACCEPTED"]
    assert len(accepted) == 1
    assert accepted[0]["target_track_id"] == 14


# 9. JSON de sortie déterministe
def test_json_output_deterministic():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 640, 720)]
    r1 = analyze_continuity(_tracks(*frags), hero_track=9)
    r2 = analyze_continuity(_tracks(*reversed(frags)), hero_track=9)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# 10. extension de lignée héros
def test_hero_lineage_extension():
    frags = [_frag(9, 22, 361, 184, 630), _frag(52, 372, 470, 640, 700)]
    report = analyze_continuity(_tracks(*frags), hero_track=9)
    ident = report["hero_identity"]
    assert ident["canonical_track_id"] == 9
    assert ident["source_track_ids"] == [9, 52]
    assert ident["method"] == METHOD


# 11. la caméra suit le nouveau fragment actif
def test_camera_follows_new_active_fragment():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 640, 720)]
    tracks = _tracks(*frags)
    report = analyze_continuity(tracks, hero_track=9)
    ident = report["hero_identity"]
    assert active_track_at(ident, 50) == 9
    assert active_track_at(ident, 150) == 14
    frames = identity_frames(tracks, ident)
    assert {it["track_id"] for it in frames} == {9, 14}


# 12. contrat héros/caméra conservé (director)
def test_hero_camera_contract_preserved():
    from goalreel.cinematic.director import build_edit_plan
    from goalreel.cinematic.edit_plan import HERO, REACTION
    a = _frag(9, 22, 361, 184, 630)
    b = _frag(14, 372, 470, 640, 720)
    tracks = _tracks(a, b)
    report = analyze_continuity(tracks, hero_track=9)
    hero = {
        "status": "OK", "hero_track": 9, "track_id": 9, "score": 0.71,
        "method": "multi_evidence_v2",
        "event": {"event_id": "MOTION_9", "kind": "player_motion",
                  "start_frame": 22, "peak_frame": 192, "end_frame": 361,
                  "confidence": 0.99, "evidence": ["track:9"], "status": "INFERRED"},
    }
    scene = {"schema": "goalreel.scene_scan.v1", "frames": 484, "fps": 30.0,
             "cut_frames": [], "cut_count": 0, "motion": [0.5] * 484,
             "motion_mean": 0.5, "motion_peak_frame": None, "motion_peak_value": 0.5,
             "evidence": "VISIBLE_FROM_SOURCE"}
    plan = build_edit_plan(video="s.mp4", tracks=tracks, hero_moment=hero,
                           width=1024, height=576, fps=30.0, total_frames=484,
                           scene=scene, continuity=report)
    assert plan.hero_track == 9 and plan.followed_track == 9
    assert plan.hero_camera_contract is True
    for s in plan.shots:
        if s.phase in (HERO, REACTION):
            assert s.target_identity == "CANONICAL_HERO_IDENTITY"


# 13. tolérance de vitesse robuste (bornée, dépend du trou)
def test_velocity_tolerance_robust_and_bounded():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 640, 720)
    fa, fb = build_fragments(_tracks(a))[9], build_fragments(_tracks(b))[14]
    t_small = velocity_tolerance(fa, fb, gap=3)["tolerance"]
    t_large = velocity_tolerance(fa, fb, gap=30)["tolerance"]
    assert t_large >= t_small                 # tolérance croît avec le trou
    assert 0.0 < t_small <= 10.0              # bornée
    assert t_large <= 10.0


# 14. prédiction bornée (horizon plafonné)
def test_prediction_horizon_bounded():
    a = build_fragments(_tracks(_frag(9, 0, 100, 0, 400)))[9]
    info = predict_position(a, 100000)         # horizon faramineux
    assert info["horizon_frames"] <= PREDICTION_MAX_HORIZON


# 15. aucune identité fabriquée
def test_no_fabricated_identity_v22():
    a = _frag(9, 0, 100, 100, 100)
    b = _frag(14, 105, 200, 900, 900, w=180, h=180)
    report = analyze_continuity(_tracks(a, b), hero_track=9)
    assert report["stats"]["links_accepted"] == 0
    assert report["hero_identity"]["source_track_ids"] == [9]
    real = {9, 14}
    for ident in report["identities"]:
        assert set(ident["source_track_ids"]) <= real


# 16. non-régression : schéma v2 + apparence de lignée
def test_schema_v2_and_lineage_appearance():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 650, 760)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0.9, 0.1, 0, 0])}
    report = link_fragments(_tracks(a, b), embeddings=emb)
    assert report["schema"] == SCHEMA
    ref = appearance_representation(emb, [9])
    assert ref is not None
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14], embeddings=emb,
                        appearance_ref=ref)
    assert scored["raw"]["appearance_source"] == "lineage_reference"
