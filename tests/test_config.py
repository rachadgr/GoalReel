"""Tests de la configuration et de la résolution de périphérique."""
from pathlib import Path

from goalreel.config import detect_device, load_settings


def test_load_settings_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("GOALREEL_"):
            monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert "detector" in s.models
    assert s.models["detector"].stage == "detection"
    # Les chemins par défaut pointent sous la racine du dépôt
    p = s.models["detector"].resolved_path()
    assert p is not None and p.is_absolute()


def test_legacy_env_override(monkeypatch):
    monkeypatch.setenv("GOALREEL_DETECTOR_WEIGHTS", "/tmp/custom.pt")
    s = load_settings()
    assert s.models["detector"].path == "/tmp/custom.pt"


def test_device_resolution(monkeypatch):
    # cpu forcé
    monkeypatch.setenv("GOALREEL_DEVICE", "cpu")
    assert detect_device("cpu") == "cpu"
    # auto -> cpu en l'absence de GPU
    assert detect_device("auto") in ("cpu", "cuda", "mps")


def test_enable_disable_flags(monkeypatch):
    monkeypatch.setenv("GOALREEL_ENABLE_DETECTOR", "false")
    s = load_settings()
    assert s.models["detector"].enabled is False
