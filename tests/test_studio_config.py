"""Tests de la configuration du Reel Studio (goalreel.studio.studio_config)."""
from __future__ import annotations

from goalreel.studio.studio_config import (
    ComfyModels,
    GenerationDefaults,
    load_studio_settings,
)


def test_defaults_are_t4_friendly():
    s = load_studio_settings({})
    assert s.comfy_url == "http://127.0.0.1:8188"
    assert s.port == 7860
    assert s.max_upload_mb == 20
    assert s.cors_origins == ["*"]
    assert s.defaults.width == 832
    assert s.defaults.height == 480
    assert s.defaults.length == 49
    assert s.defaults.fps == 16.0
    assert s.defaults.steps == 20
    assert s.defaults.cfg == 5.0


def test_env_override_comfy_and_http():
    s = load_studio_settings(
        {
            "COMFY_URL": "http://10.0.0.5:9000/",
            "HOST": "127.0.0.1",
            "PORT": "8001",
            "MAX_UPLOAD_MB": "50",
            "CORS_ORIGINS": "https://a.com, https://b.com",
        }
    )
    assert s.comfy_url == "http://10.0.0.5:9000"  # slash final retiré
    assert s.host == "127.0.0.1"
    assert s.port == 8001
    assert s.max_upload_mb == 50
    assert s.max_upload_bytes == 50 * 1024 * 1024
    assert s.cors_origins == ["https://a.com", "https://b.com"]


def test_goalreel_prefix_takes_priority():
    s = load_studio_settings(
        {
            "GOALREEL_STUDIO_COMFY_URL": "http://prefixed:1234",
            "COMFY_URL": "http://fallback:8188",
        }
    )
    assert s.comfy_url == "http://prefixed:1234"


def test_model_filenames_match_target_environment():
    s = load_studio_settings({})
    assert s.models.diffusion == "wan2.2_ti2v_5B_fp16.safetensors"
    assert s.models.vae == "wan2.2_vae.safetensors"
    assert s.models.text_encoder == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"


def test_model_env_override():
    s = load_studio_settings({"WAN_DIFFUSION_MODEL": "custom.safetensors"})
    assert s.models.diffusion == "custom.safetensors"


def test_public_dict_has_no_secrets():
    s = load_studio_settings({})
    d = s.public_dict()
    assert "comfy_url" in d and "models" in d and "defaults" in d
    # Aucune clé sensible ne doit apparaître.
    flat = str(d).lower()
    for token in ("token", "secret", "password", "api_key"):
        assert token not in flat


def test_invalid_numbers_fall_back_to_default():
    s = load_studio_settings({"PORT": "not-a-number", "MAX_UPLOAD_MB": "abc"})
    assert s.port == 7860
    assert s.max_upload_mb == 20


def test_ensure_dirs(tmp_path):
    from goalreel.studio.studio_config import StudioSettings

    s = StudioSettings(
        comfy_url="http://x",
        host="0.0.0.0",
        port=7860,
        max_upload_mb=20,
        cors_origins=["*"],
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        job_timeout_s=10,
        poll_interval_s=1.0,
        max_retries=1,
        retry_backoff_s=0.1,
        http_timeout_s=5.0,
    )
    s.ensure_dirs()
    assert s.upload_dir.is_dir() and s.output_dir.is_dir()


def test_comfy_models_and_defaults_constructible():
    assert ComfyModels().diffusion.endswith(".safetensors")
    assert GenerationDefaults().seed == 123456789
