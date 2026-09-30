"""Tests de régression — Cinematic Director V2 (plan de montage).

Verrouille les garanties produit et de vérité du director :

  * déterminisme : mêmes preuves => plan identique (JSON stable) ;
  * partition du temps : plans contigus couvrant exactement [0, total_frames) ;
  * ordre des phases conforme à PHASE_ORDER ;
  * exigence de preuve : aucune phase sans événement/preuve soutenante ;
  * préservation du héros : hero_track == followed_track ;
  * couplage héros <-> caméra : cibles = centres réellement suivis ;
  * bornes de vitesse : tout facteur dans [SPEED_MIN, SPEED_MAX] ;
  * durée : sum(out_frames) == total_frames, facteurs effectifs bornés ;
  * aucun événement fabriqué ; sans héros => aucun beat héros ;
  * phases absentes documentées avec raison ;
  * RIFE jamais revendiqué (mode réel TEMPORAL_RESAMPLE).
"""
from __future__ import annotations

import json

import pytest

from goalreel.cinematic.director import (
    build_edit_plan,
    dominant_track_at,
    hero_action_onset,
    hero_beats,
    hero_speed_profile,
    resolve_timeline,
)
from goalreel.cinematic.edit_plan import (
    ACTION,
    ANTICIPATION,
    FINAL_HERO,
    HERO,
    HOOK,
    MIN_SHOT_FRAMES,
    OUTRO,
    PHASE_ORDER,
    REACTION,
    SPEED_MAX,
    SPEED_MIN,
    SUPPORTED_TRANSITIONS,
)
from goalreel.cinematic.render_plan import (
    build_frame_map,
    crop_window,
    plan_validation,
    resample_index,
    target_at,
)
from goalreel.cinematic.speed_design import factors_are_bounded, speed_curve
from goalreel.cinematic.transitions import is_supported_transition


# --- fixtures synthétiques (preuve minimale, aucun modèle requis) ----------
def _track(tid, start, end, cx0, cx1, w=20.0, h=50.0, cy=380.0):
    n = end - start + 1
    out = []
    for i in range(n):
        cx = cx0 + (cx1 - cx0) * (i / max(1, n - 1))
        out.append({"frame": start + i,
                    "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return out


def _scene(frames=484, cuts=None, peak=None):
    n = frames
    mot = [0.5] * n
    if peak is not None:
        mot[peak] = 20.0
    return {"schema": "goalreel.scene_scan.v1", "frames": n, "fps": 30.0,
            "cut_frames": list(cuts or []), "cut_count": len(cuts or []),
            "motion": mot, "motion_mean": float(sum(mot) / len(mot)),
            "motion_peak_frame": peak, "motion_peak_value": 20.0 if peak else 0.5,
            "evidence": "VISIBLE_FROM_SOURCE"}


def _hero(track_id, peak=318):
    return {"schema": "goalreel.hero_moment.v1", "status": "OK",
            "hero_track": track_id, "track_id": track_id, "score": 0.71,
            "method": "multi_evidence_v2",
            "event": {"event_id": "MOTION_9", "kind": "player_motion",
                      "start_frame": 22, "peak_frame": peak, "end_frame": 361,
                      "confidence": 0.99,
                      "evidence": [f"track:{track_id}", "displacement:450.11"],
                      "status": "INFERRED"}}


def _tracks_with_hero():
    return {2: _track(2, 0, 361, 281, 827),
            7: _track(7, 14, 352, 316, 420),
            9: _track(9, 22, 361, 184, 630)}


def _plan(**kw):
    tracks = kw.pop("tracks", None)
    if tracks is None:
        tracks = _tracks_with_hero()
    hero = kw.pop("hero", None)
    if hero is None:
        hero = _hero(9)
    kw.setdefault("video", "source.mp4")
    kw.setdefault("width", 1024)
    kw.setdefault("height", 576)
    kw.setdefault("fps", 30.0)
    kw.setdefault("total_frames", 484)
    kw.setdefault("scene", _scene())
    return build_edit_plan(tracks=tracks, hero_moment=hero, **kw)


# --- déterminisme ----------------------------------------------------------
def test_edit_plan_is_deterministic():
    a = _plan().to_dict()
    b = _plan().to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_determinism_independent_of_insertion_order():
    tracks = _tracks_with_hero()
    shuffled = {k: tracks[k] for k in (9, 2, 7)}
    a = _plan(tracks=tracks).to_dict()
    b = _plan(tracks=shuffled).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- partition / ordre / durée --------------------------------------------
def test_shots_partition_time_exactly():
    plan = _plan()
    expected = 0
    for s in plan.shots:
        assert int(s.start_frame) == expected
        assert s.end_frame >= s.start_frame
        expected = int(s.end_frame) + 1
    assert expected == plan.total_frames


def test_phase_order_is_respected():
    plan = _plan()
    idx = [PHASE_ORDER.index(s.phase) for s in plan.shots]
    assert idx == sorted(idx)


def test_core_phases_generated_on_full_evidence():
    plan = _plan(scene=_scene(cuts=[332], peak=442))
    for ph in (HOOK, HERO, OUTRO):
        assert ph in plan.phases_present, ph


def test_duration_preserved_and_factors_bounded():
    plan = _plan()
    assert plan.duration_preserved is True
    assert plan.time_map["total_out_frames"] == plan.total_frames
    for sid, info in plan.time_map["shots"].items():
        assert SPEED_MIN - 1e-6 <= info["effective_factor"] <= SPEED_MAX + 1e-6
        assert info["in_bounds"] is True


def test_no_shot_shorter_than_minimum():
    assert all(s.duration_frames >= MIN_SHOT_FRAMES for s in _plan().shots)


# --- héros / caméra --------------------------------------------------------
def test_hero_track_equals_followed_track():
    plan = _plan()
    assert plan.hero_track == 9
    assert plan.followed_track == plan.hero_track
    assert plan.hero_camera_contract is True


def test_hero_centric_shots_follow_hero_track():
    plan = _plan()
    for s in plan.shots:
        if s.phase in (ANTICIPATION, ACTION, HERO, REACTION, FINAL_HERO):
            assert s.selected_track == 9


def test_hero_targets_come_from_real_track():
    tracks = _tracks_with_hero()
    plan = _plan(tracks=tracks, scene=_scene(cuts=[332], peak=442))
    hero = next(s for s in plan.shots if s.phase == HERO)
    assert hero.targets
    real = {f["frame"]: (f["bbox"][0] + f["bbox"][2]) / 2 for f in tracks[9]}
    for tgt in hero.targets:
        assert tgt["track_id"] == 9
        if not tgt["held"]:
            assert tgt["cx"] == pytest.approx(real[tgt["frame"]], abs=1e-3)


def test_no_hero_evidence_yields_no_hero_beats():
    plan = _plan(hero={"status": "UNKNOWN"})
    assert plan.hero_track is None
    assert plan.hero_camera_contract is False
    assert all(s.phase not in (HERO, ANTICIPATION, FINAL_HERO)
               for s in plan.shots)
    assert any(f["reason"].startswith("NO_HERO") for f in plan.fallbacks)


def test_no_invented_coordinates_when_hero_absent():
    tracks = {2: _track(2, 0, 200, 100, 300)}
    plan = _plan(tracks=tracks, hero=_hero(999))
    for s in plan.shots:
        for tgt in s.targets:
            assert tgt["track_id"] in {2}


def test_camera_targets_cover_shot_interval_when_present():
    checked = 0
    for s in _plan().shots:
        if not s.targets:
            continue
        frames = [t["frame"] for t in s.targets]
        assert frames == list(range(s.start_frame, s.end_frame + 1))
        checked += 1
    assert checked > 0


# --- preuve / fabrication --------------------------------------------------
def test_missing_phases_documented_with_reason():
    plan = _plan(tracks={2: _track(2, 0, 120, 100, 200)}, hero=_hero(2))
    present = set(plan.phases_present)
    for e in plan.phases_absent:
        assert e["phase"] not in present
        assert e["reason"]
    assert {e["phase"] for e in plan.phases_absent} == \
        {p for p in PHASE_ORDER if p not in present}


def test_no_fabricated_event_ids():
    plan = _plan()
    for s in plan.shots:
        for ev in s.evidence:
            if ev.startswith("hero_event:"):
                assert ev.split(":", 1)[1] == plan.hero_event_id


def test_ball_evidence_never_fabricated():
    plan = _plan(ball_detections=[])
    assert plan.evidence_used["ball_evidence_available"] is False
    assert plan.evidence_used["ball_detections"] == 0


def test_real_ball_evidence_recorded():
    balls = [{"frame": 43, "bbox": [880, 500, 912, 519], "confidence": 0.41},
             {"frame": 442, "bbox": [530, 175, 553, 198], "confidence": 0.69}]
    plan = _plan(ball_detections=balls)
    assert plan.evidence_used["ball_evidence_available"] is True
    assert plan.evidence_used["ball_detections"] == 2


# --- vitesse ---------------------------------------------------------------
def test_speed_curves_bounded_for_every_shot():
    for s in _plan().shots:
        lo = float(s.speed_curve["start_factor"])
        hi = float(s.speed_curve["end_factor"])
        assert SPEED_MIN - 1e-9 <= lo <= SPEED_MAX + 1e-9
        assert SPEED_MIN - 1e-9 <= hi <= SPEED_MAX + 1e-9


def test_slowdown_only_with_event_evidence():
    c_ok = speed_curve(HERO, event_supported=True, confidence=0.9)
    c_no = speed_curve(HERO, event_supported=False, confidence=0.9)
    assert c_ok.end_factor < 1.0
    assert c_no.start_factor == 1.0 and c_no.end_factor == 1.0
    assert c_no.reason == "NO_SPEED_EVIDENCE"
    assert factors_are_bounded(c_ok) and factors_are_bounded(c_no)


def test_speed_design_never_claims_rife_without_rife():
    c = speed_curve(HERO, event_supported=True, confidence=0.9,
                    rife_available=False)
    assert c.fallback == "TEMPORAL_RESAMPLE_NO_RIFE"
    assert speed_curve(HERO, True, 0.9, rife_available=True).fallback is None


def test_plan_claims_never_claim_rife_or_novel_view():
    c = _plan().claims
    assert c["rife_used"] is False
    assert c["interpolation_mode"] == "TEMPORAL_RESAMPLE_NO_RIFE"
    assert c["novel_view"] == "NOT_USED"
    assert c["generated_pixels"] is False
    assert c["real_source_only"] is True


# --- transitions -----------------------------------------------------------
def test_only_supported_transitions_used():
    for s in _plan().shots:
        assert is_supported_transition(s.transition_in)
        assert is_supported_transition(s.transition_out)
        assert s.transition_in in SUPPORTED_TRANSITIONS
        assert s.transition_out in SUPPORTED_TRANSITIONS


def test_opening_and_closing_transitions_frame_the_reel():
    plan = _plan()
    assert plan.shots[0].transition_in == "FADE_FROM_BLACK"
    assert plan.shots[-1].transition_out == "FADE_TO_BLACK"


def test_real_source_cut_recorded_as_measured_boundary():
    plan = _plan(scene=_scene(cuts=[332], peak=442))
    on_cut = [s for s in plan.shots if s.boundary == "REAL_SOURCE_CUT"]
    assert on_cut, "aucune frontière de plan réelle détectée"
    for s in on_cut:
        assert any(ev == "source_cut_boundary" for ev in s.evidence)
    assert plan.evidence_used["real_cut_frames"] == [332]


# --- helpers de preuve -----------------------------------------------------
def test_resolve_timeline_partitions_and_merges():
    segs = resolve_timeline([(0, 5, HOOK), (6, 100, "BUILD_UP"), (101, 200, HERO)],
                            total_frames=201, min_frames=12)
    assert segs[0][0] == 0 and segs[-1][1] == 200
    for (_, e1, _), (s2, _, _) in zip(segs, segs[1:]):
        assert e1 + 1 == s2


def test_resolve_timeline_empty_cases():
    assert resolve_timeline([], 100) == []
    assert resolve_timeline([(0, 9, HOOK)], 0) == []


def test_hero_speed_profile_and_beats_factual():
    prof = hero_speed_profile(_tracks_with_hero()[9])
    assert prof and all({"frame", "speed"} <= set(p) for p in prof)
    assert hero_beats(prof)["insufficient_evidence"] is False


def test_hero_beats_insufficient_evidence_honest():
    assert hero_beats([])["insufficient_evidence"] is True
    assert hero_beats([{"frame": 1, "speed": 0.0}])["primary"] is None


def test_hero_action_onset_before_peak():
    prof = hero_speed_profile(_tracks_with_hero()[9])
    peak = max(p["frame"] for p in prof)
    onset = hero_action_onset(prof, peak)
    if onset is not None:
        assert onset < peak


def test_dominant_track_prefers_person_sized_subject():
    tracks = {1: _track(1, 0, 31, 100, 140),
              2: [{"frame": f, "bbox": [0, 0, 1000, 560]} for f in range(32)]}
    assert dominant_track_at(tracks, 0) == 1
    assert dominant_track_at(tracks, 0, prefer_person_sized=False) == 2


# --- renderer plan-aware (helpers purs, sans FFmpeg) -----------------------
def test_crop_window_stays_inside_source():
    for frac in (0.5, 0.72, 1.0, 1.5):
        y0, ch = crop_window(576, frac)
        assert 0 <= y0 <= 576 - ch
        assert 2 <= ch <= 576


def test_resample_index_monotonic_no_freeze():
    out_n, src_n = 78, 61
    idx = [resample_index(k, out_n, src_n)[0] for k in range(out_n)]
    assert idx[0] == 0 and idx[-1] <= src_n - 1
    assert all(b >= a for a, b in zip(idx, idx[1:]))
    assert len(set(idx)) > 1


def test_resample_index_centred_sampling():
    i0, i1, frac = resample_index(0, 10, 10)
    assert i0 == 0 and frac == 0.0


def test_build_frame_map_preserves_total_and_progression():
    plan = _plan()
    fm = build_frame_map(plan)
    assert len(fm) == plan.total_frames
    assert fm[0]["out_frame"] == 0
    assert fm[-1]["out_frame"] == plan.total_frames - 1
    per_shot: dict[str, list[int]] = {}
    for e in fm:
        per_shot.setdefault(e["shot_id"], []).append(e["src_frame"])
    for seq in per_shot.values():
        assert all(b >= a for a, b in zip(seq, seq[1:]))


def test_blend_only_when_shot_not_accelerated():
    plan = _plan()
    alloc = plan.time_map["shots"]
    per_shot: dict[str, list] = {}
    for e in build_frame_map(plan):
        per_shot.setdefault(e["shot_id"], []).append(e)
    checked = 0
    for sid, entries in per_shot.items():
        if alloc[sid]["src_frames"] > alloc[sid]["out_frames"]:
            assert all(e["blend"] == 0.0 for e in entries), sid
            checked += 1
        else:
            assert all(0.0 <= e["blend"] < 1.0 for e in entries), sid
    assert checked >= 1


def test_plan_validation_ok_then_detects_gap():
    plan = _plan()
    assert plan_validation(plan, plan.time_map["shots"])["status"] == "OK"
    plan.shots[1].start_frame += 5
    bad = plan_validation(plan, plan.time_map["shots"])
    assert bad["status"] == "QC_REJECTED"
    assert any("timeline_gap" in e for e in bad["errors"])


def test_target_at_holds_last_known_real_position():
    plan = _plan()
    hero = next(s for s in plan.shots if s.phase == HERO)
    t = target_at(hero, hero.targets[0]["frame"] - 50)
    assert t is not None and t[0] == pytest.approx(hero.targets[0]["cx"], abs=1e-3)
