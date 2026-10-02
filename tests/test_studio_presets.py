"""Tests du système de presets football."""
from __future__ import annotations

from goalreel.studio import presets


EXPECTED_IDS = {
    "cinematic_football",
    "goal_celebration",
    "dribble",
    "sprint",
    "shot_on_goal",
    "goalkeeper_save",
    "player_introduction",
    "slow_motion_hero",
}


def test_all_expected_presets_exist():
    assert set(presets.preset_ids()) == EXPECTED_IDS


def test_default_preset_is_cinematic_football():
    assert presets.DEFAULT_PRESET_ID == "cinematic_football"
    assert presets.get_preset(None).id == "cinematic_football"
    assert presets.get_preset("does_not_exist").id == "cinematic_football"


def test_each_preset_has_positive_and_negative():
    for p in presets.list_presets():
        assert p.positive.strip()
        assert p.negative.strip()
        assert p.label
        assert p.description


def test_presets_include_realism_style_base():
    for p in presets.list_presets():
        assert "professional stadium" in p.positive
        assert "realistic" in p.positive.lower()


def test_presets_exclude_artifacts_base():
    for p in presets.list_presets():
        neg = p.negative.lower()
        for token in ("extra limbs", "duplicate players", "flicker", "watermark"):
            assert token in neg


def test_build_prompts_uses_preset():
    pos, neg = presets.build_prompts("sprint")
    assert "sprint" in pos.lower() or "running" in pos.lower()
    assert "extra limbs" in neg


def test_build_prompts_override_keeps_base():
    pos, neg = presets.build_prompts(
        "dribble", positive_override="rainy night match", negative_override="fog"
    )
    assert "rainy night match" in pos
    assert "professional stadium" in pos  # base conservée
    assert "fog" in neg
    assert "extra limbs" in neg  # base négative conservée


def test_to_dict_roundtrip():
    d = presets.get_preset("goal_celebration").to_dict()
    assert set(d.keys()) == {"id", "label", "description", "positive", "negative"}
