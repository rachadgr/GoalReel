"""Tests déterministes — Subject Continuity V2.1 (identité canonique).

Vérrouille les garanties de la couche de continuité :

  1. continuité même-track            (identité dégénérée, aucune fusion)
  2. association sur trou court
  3. continuité spatiale              (saut de position → rejet)
  4. continuité de vitesse            (mouvement incohérent → rejet)
  5. similarité d'apparence OSNet
  6. association sur preuves combinées
  7. rejet de fausse fusion
  8. rejet de trou long
  9. rejet d'apparence contradictoire
 10. départage déterministe
 11. construction de lignée héros
 12. plusieurs tracks bruts → une identité canonique
 13. la caméra suit le fragment héros actif
 14. contrat héros/caméra conservé
 15. fallback fragment non résolu
 16. fallback OSNet manquant
 17. fallback SAM2 manquant
 18. aucune identité fabriquée
 19. aucun événement fabriqué
 20. sortie JSON déterministe
 21. phases post-héros uniquement si la preuve existe (couplage director)
 22. le comportement du Director V2 reste intact
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from goalreel.cinematic.director import build_edit_plan
from goalreel.cinematic.edit_plan import (
    FINAL_HERO,
    HERO,
    HOOK,
    OUTRO,
    REACTION,
)
from goalreel.tracking.continuity import (
    ACCEPT_THRESHOLD,
    FALLBACK_NO_APPEARANCE,
    FALLBACK_UNRESOLVED,
    active_track_at,
    analyze_continuity,
    build_fragments,
    hero_identity,
    identity_containing,
    identity_frames,
    link_fragments,
    score_link,
)


# ---------------------------------------------------------------------------
# Fixtures synthétiques (preuve minimale, aucun modèle requis)
# ---------------------------------------------------------------------------
def _frag(tid, start, end, cx0, cx1, cy=380.0, w=20.0, h=50.0, conf=0.9):
    """Fragment linéaire déterministe : centre qui va de ``cx0`` à ``cx1``."""
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


# ---------------------------------------------------------------------------
# 1. continuité même-track
# ---------------------------------------------------------------------------
def test_same_track_identity_is_degenerate():
    tracks = _tracks(_frag(9, 0, 100, 200, 600))
    report = analyze_continuity(tracks, hero_track=9)
    ident = report["hero_identity"]
    assert ident["source_track_ids"] == [9]
    assert ident["canonical_track_id"] == 9
    assert ident["links"] == []
    assert report["stats"]["tracks_merged"] == 0


# ---------------------------------------------------------------------------
# 2. association sur trou court
# ---------------------------------------------------------------------------
def test_short_gap_association_accepted():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)     # même direction, trou de 11 frames
    report = analyze_continuity(_tracks(a, b), hero_track=9)
    ident = report["hero_identity"]
    assert ident["source_track_ids"] == [9, 14]
    assert report["stats"]["links_accepted"] >= 1


# ---------------------------------------------------------------------------
# 3. continuité spatiale
# ---------------------------------------------------------------------------
def test_spatial_continuity_rejects_far_jump():
    a = _frag(9, 0, 100, 100, 100)        # reste à x=100
    b = _frag(14, 105, 200, 900, 900)     # réapparaît à x=900
    report = link_fragments(_tracks(a, b))
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["decision"] == "REJECTED"
    assert link["rejection_reason"] in ("SPATIAL_VELOCITY_CONFLICT", "BELOW_THRESHOLD")
    # aucune fuite de clé interne (les clés privées commencent par "_")
    assert not any(str(k).startswith("_") for k in link)


# ---------------------------------------------------------------------------
# 4. continuité de vitesse
# ---------------------------------------------------------------------------
def test_velocity_continuity_flag():
    a = _frag(9, 0, 100, 0, 400)          # vitesse +4 px/frame vers la droite
    b = _frag(14, 104, 200, 400, 370)     # repart vers la gauche (incohérent)
    scored = score_link(build_fragments(_tracks(a))[9], build_fragments(_tracks(b))[14])
    assert scored["raw"]["velocity_delta"] > 0.0
    assert scored["terms"]["velocity"]["value"] < 1.0


# ---------------------------------------------------------------------------
# 5. similarité d'apparence OSNet
# ---------------------------------------------------------------------------
def test_osnet_appearance_similarity_used():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0.98, 0.05, 0, 0])}
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14], embeddings=emb)
    assert scored["raw"]["appearance_available"] is True
    assert scored["raw"]["appearance_cosine"] > 0.9
    assert scored["terms"]["appearance"]["value"] > 0.9


# ---------------------------------------------------------------------------
# 6. association sur preuves combinées
# ---------------------------------------------------------------------------
def test_combined_evidence_association():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0.99, 0.02, 0, 0])}
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14], embeddings=emb)
    assert scored["score"] > ACCEPT_THRESHOLD
    for k in ("temporal", "spatial", "velocity", "geometry", "persistence", "appearance"):
        assert k in scored["terms"]
        assert scored["terms"][k]["contribution"] is not None


# ---------------------------------------------------------------------------
# 7. rejet de fausse fusion
# ---------------------------------------------------------------------------
def test_false_merge_rejected_by_geometry_and_motion():
    a = _frag(9, 0, 40, 300, 310, w=18, h=46)     # joueur normal
    b = _frag(50, 44, 90, 900, 900, w=400, h=300)  # blob large très à droite
    report = link_fragments(_tracks(a, b))
    link = next((e for e in report["links"]
                 if e["source_track_id"] == 9 and e["target_track_id"] == 50), None)
    assert link is not None and link["decision"] == "REJECTED"


# ---------------------------------------------------------------------------
# 8. rejet de trou long
# ---------------------------------------------------------------------------
def test_long_gap_rejected():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 300, 400, 610, 700)     # trou de ~199 frames
    report = link_fragments(_tracks(a, b))
    # Aucun lien candidat n'est même évalué (hors fenêtre bornée).
    pairs = {(e["source_track_id"], e["target_track_id"]) for e in report["links"]}
    assert (9, 14) not in pairs
    assert report["stats"]["links_accepted"] == 0


# ---------------------------------------------------------------------------
# 9. rejet d'apparence contradictoire
# ---------------------------------------------------------------------------
def test_contradictory_appearance_rejected():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)     # spatialement plausible…
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0, 1, 0, 0])}   # cosine == 0 (contradiction)
    scored = score_link(build_fragments(_tracks(a))[9],
                        build_fragments(_tracks(b))[14], embeddings=emb)
    assert scored["raw"]["appearance_cosine"] == pytest.approx(0.0, abs=1e-6)
    report = link_fragments(_tracks(a, b), embeddings=emb)
    link = next(e for e in report["links"]
                if e["source_track_id"] == 9 and e["target_track_id"] == 14)
    assert link["decision"] == "REJECTED"
    assert link["rejection_reason"] == "CONTRADICTORY_APPEARANCE"


# ---------------------------------------------------------------------------
# 10. départage déterministe
# ---------------------------------------------------------------------------
def test_deterministic_tie_breaking_and_order_independence():
    a = _frag(9, 0, 100, 200, 600)
    b1 = _frag(14, 112, 200, 610, 700)
    b2 = _frag(21, 112, 200, 610, 700)    # candidats identiques → départage stable
    tracks = _tracks(a, b1, b2)
    r1 = analyze_continuity(tracks, hero_track=9)
    r2 = analyze_continuity({21: tracks[21], 14: tracks[14], 9: tracks[9]}, hero_track=9)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    accepted = [e for e in r1["links"] if e["decision"] == "ACCEPTED"]
    assert len(accepted) == 1                       # un seul successeur pour la source 9
    assert accepted[0]["target_track_id"] == 14      # plus petit id (départage total)


def test_acceptance_is_injective():
    # deux sources ne peuvent pas fusionner vers la même cible
    a1 = _frag(9, 0, 100, 200, 600)
    a2 = _frag(10, 0, 100, 210, 610)
    b = _frag(14, 112, 200, 620, 700)
    report = link_fragments(_tracks(a1, a2, b))
    accepted = [e for e in report["links"] if e["decision"] == "ACCEPTED"]
    tgt = [e["target_track_id"] for e in accepted]
    assert len(tgt) == len(set(tgt))


# ---------------------------------------------------------------------------
# 11. construction de lignée héros
# ---------------------------------------------------------------------------
def test_hero_lineage_construction():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 510, 640),
             _frag(21, 214, 300, 650, 800)]
    report = analyze_continuity(_tracks(*frags), hero_track=9)
    ident = report["hero_identity"]
    assert ident["source_track_ids"] == [9, 14, 21]
    assert ident["canonical_track_id"] == 9
    assert len(ident["frame_ranges"]) == 3
    assert len(ident["links"]) == 2
    assert ident["confidence"] > 0.0
    assert ident["fallback"] is None


# ---------------------------------------------------------------------------
# 12. plusieurs tracks bruts → une identité canonique
# ---------------------------------------------------------------------------
def test_multiple_raw_tracks_become_one_identity():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 510, 640)]
    report = analyze_continuity(_tracks(*frags), hero_track=9)
    containing = identity_containing(report, 14)
    assert containing is not None
    assert containing["source_track_ids"] == [9, 14]
    assert report["stats"]["identity_count"] == 1
    assert report["stats"]["tracks_merged"] == 1


# ---------------------------------------------------------------------------
# 13. la caméra suit le fragment héros actif
# ---------------------------------------------------------------------------
def test_camera_follows_active_hero_fragment():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 510, 640)]
    tracks = _tracks(*frags)
    report = analyze_continuity(tracks, hero_track=9)
    ident = report["hero_identity"]
    assert active_track_at(ident, 50) == 9
    assert active_track_at(ident, 150) == 14
    assert active_track_at(ident, 5000) is None
    # frames réelles concaténées : chaque item porte son fragment d'origine
    frames = identity_frames(tracks, ident)
    used_ids = {it["track_id"] for it in frames}
    assert used_ids == {9, 14}
    assert frames == sorted(frames, key=lambda it: it["frame"])


# ---------------------------------------------------------------------------
# 14. contrat héros/caméra conservé (director consomme la lignée)
# ---------------------------------------------------------------------------
def test_hero_camera_contract_preserved_with_lineage():
    a = _frag(9, 22, 361, 184, 630)
    b = _frag(14, 372, 470, 630, 700)
    tracks = _tracks(a, b)
    report = analyze_continuity(tracks, hero_track=9)
    hero = {
        "status": "OK", "hero_track": 9, "track_id": 9, "score": 0.71,
        "method": "multi_evidence_v2",
        "event": {"event_id": "MOTION_9", "kind": "player_motion",
                  "start_frame": 22, "peak_frame": 192, "end_frame": 361,
                  "confidence": 0.99, "evidence": ["track:9"], "status": "INFERRED"},
    }
    plan = build_edit_plan(video="s.mp4", tracks=tracks, hero_moment=hero,
                           width=1024, height=576, fps=30.0, total_frames=484,
                           scene=_scene(), continuity=report)
    assert plan.hero_track == 9
    assert plan.followed_track == 9
    assert plan.hero_camera_contract is True
    assert plan.hero_identity is not None
    assert plan.hero_identity["source_track_ids"] == [9, 14]
    for s in plan.shots:
        if s.phase in (HERO, REACTION):
            assert s.target_identity == "CANONICAL_HERO_IDENTITY"


# ---------------------------------------------------------------------------
# 15. fallback fragment non résolu
# ---------------------------------------------------------------------------
def test_unresolved_fragment_fallback_recorded():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)
    emb = {9: _emb([1, 0, 0, 0]), 14: _emb([0, 1, 0, 0])}   # apparence contradictoire
    report = analyze_continuity(_tracks(a, b), embeddings=emb, hero_track=9)
    ident = report["hero_identity"]
    assert ident["source_track_ids"] == [9]
    assert ident["fallback"] == FALLBACK_UNRESOLVED
    assert ident["fallback_detail"]["target_track_id"] == 14
    assert ident["fallback_detail"]["reason"] == "CONTRADICTORY_APPEARANCE"


# ---------------------------------------------------------------------------
# 16. fallback OSNet manquant (apparence indisponible, pas de simulation)
# ---------------------------------------------------------------------------
def test_missing_osnet_fallback_fails_soft():
    a = _frag(9, 0, 100, 200, 600)
    b = _frag(14, 112, 200, 610, 700)
    report = link_fragments(_tracks(a, b), embeddings=None)
    assert report["appearance"]["available"] is False
    assert report["appearance"]["fallback"] == FALLBACK_NO_APPEARANCE
    # la fusion reste possible via les autres preuves (échec en douceur)
    assert report["stats"]["links_accepted"] >= 1


# ---------------------------------------------------------------------------
# 17. fallback SAM2 manquant
# ---------------------------------------------------------------------------
def test_missing_sam2_reported_unavailable():
    a = _frag(9, 0, 100, 200, 600)
    report = link_fragments(_tracks(a))
    assert report["sam2"]["available"] is False
    assert report["sam2"]["fallback"] == "SAM2_EVIDENCE_UNAVAILABLE"
    assert "identity oracle" in report["sam2"]["note"]


# ---------------------------------------------------------------------------
# 18. aucune identité fabriquée
# ---------------------------------------------------------------------------
def test_no_fabricated_identity():
    # deux fragments incompatibles : aucune fusion, aucune identité inventée
    a = _frag(9, 0, 100, 100, 100)
    b = _frag(14, 105, 200, 900, 900, w=180, h=180)   # blob large + saut énorme
    report = analyze_continuity(_tracks(a, b), hero_track=9)
    assert report["stats"]["links_accepted"] == 0
    assert report["hero_identity"]["source_track_ids"] == [9]
    ids = {t["track_id"] for t in report["raw_tracks"]}
    assert ids == {9, 14}
    # chaque identité référence uniquement des tracks réels
    real = {int(t) for t in _tracks(a, b)}
    for ident in report["identities"]:
        assert set(ident["source_track_ids"]) <= real


# ---------------------------------------------------------------------------
# 19. aucun événement fabriqué
# ---------------------------------------------------------------------------
def test_no_fabricated_event():
    a = _frag(9, 0, 100, 200, 600)
    report = link_fragments(_tracks(a))
    # Aucun champ d'événement football n'est produit par la continuité (le mot
    # "goalreel" contient "goal" : on n'inspecte que les CLÉS, pas le schéma).
    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from _keys(it)
    keys = {str(k).lower() for k in _keys(report)}
    for banned in ("goal_event", "pass", "celebration", "possession",
                   "goal_scored", "shot_event", "player_identity"):
        assert banned not in keys


# ---------------------------------------------------------------------------
# 20. sortie JSON déterministe
# ---------------------------------------------------------------------------
def test_deterministic_json_output():
    frags = [_frag(9, 0, 100, 200, 500), _frag(14, 112, 200, 510, 640),
             _frag(21, 214, 300, 650, 800)]
    r1 = analyze_continuity(_tracks(*frags), hero_track=9)
    r2 = analyze_continuity(_tracks(*reversed(frags)), hero_track=9)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# ---------------------------------------------------------------------------
# 21. phases post-héros uniquement si la preuve existe
# ---------------------------------------------------------------------------
def test_post_hero_phases_unlocked_by_continuation():
    # le héros brut 9 s'arrête à 361 ; le fragment 14 continue réellement après.
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
    plan = build_edit_plan(video="s.mp4", tracks=tracks, hero_moment=hero,
                           width=1024, height=576, fps=30.0, total_frames=484,
                           scene=_scene(), continuity=report)
    assert REACTION in plan.phases_present
    # la caméra du plan REACTION suit bien l'identité canonique héros
    reaction = next((s for s in plan.shots if s.phase == REACTION), None)
    assert reaction is not None
    assert reaction.selected_track == 9
    assert reaction.target_identity == "CANONICAL_HERO_IDENTITY"


def test_no_post_hero_phase_without_continuation_evidence():
    # héros isolé qui s'arrête : aucune continuation réelle → pas d'invention.
    a = _frag(9, 22, 360, 184, 630)
    tracks = _tracks(a)
    report = analyze_continuity(tracks, hero_track=9)
    hero = {
        "status": "OK", "hero_track": 9, "track_id": 9, "score": 0.71,
        "method": "multi_evidence_v2",
        "event": {"event_id": "MOTION_9", "kind": "player_motion",
                  "start_frame": 22, "peak_frame": 192, "end_frame": 360,
                  "confidence": 0.99, "evidence": ["track:9"], "status": "INFERRED"},
    }
    plan = build_edit_plan(video="s.mp4", tracks=tracks, hero_moment=hero,
                           width=1024, height=576, fps=30.0, total_frames=484,
                           scene=_scene(), continuity=report)
    # aucune extension fabriquée : le plan reste honnête
    assert plan.hero_identity["source_track_ids"] == [9]
    assert all(s.phase != HERO or s.selected_track == 9 for s in plan.shots)


# ---------------------------------------------------------------------------
# 22. le comportement du Director V2 reste intact (sans continuité)
# ---------------------------------------------------------------------------
def _scene(frames=484, cuts=None, peak=None):
    mot = [0.5] * frames
    if peak is not None:
        mot[peak] = 20.0
    return {"schema": "goalreel.scene_scan.v1", "frames": frames, "fps": 30.0,
            "cut_frames": list(cuts or []), "cut_count": len(cuts or []),
            "motion": mot, "motion_mean": float(sum(mot) / len(mot)),
            "motion_peak_frame": peak, "motion_peak_value": 20.0 if peak else 0.5,
            "evidence": "VISIBLE_FROM_SOURCE"}


def test_director_v2_behavior_intact_without_continuity():
    tracks = {2: _frag(2, 0, 361, 281, 827),
              7: _frag(7, 14, 352, 316, 420),
              9: _frag(9, 22, 361, 184, 630)}
    hero = {
        "status": "OK", "hero_track": 9, "track_id": 9, "score": 0.71,
        "method": "multi_evidence_v2",
        "event": {"event_id": "MOTION_9", "kind": "player_motion",
                  "start_frame": 22, "peak_frame": 192, "end_frame": 361,
                  "confidence": 0.99, "evidence": ["track:9"], "status": "INFERRED"},
    }
    plan = build_edit_plan(video="s.mp4", tracks=tracks, hero_moment=hero,
                           width=1024, height=576, fps=30.0, total_frames=484,
                           scene=_scene(cuts=[332], peak=442))
    # contrat, partition et phases héros historiques inchangés
    assert plan.hero_track == 9 and plan.followed_track == 9
    assert plan.hero_camera_contract is True
    assert HERO in plan.phases_present and HOOK in plan.phases_present
    assert plan.duration_preserved is True
    expected = 0
    for s in plan.shots:
        assert int(s.start_frame) == expected
        expected = int(s.end_frame) + 1
    assert expected == plan.total_frames
