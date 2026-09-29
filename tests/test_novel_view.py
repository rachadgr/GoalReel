"""Tests novel-view : provider, fallback, absence de backend."""
import pytest

from goalreel.generation.novel_view.planner import NovelViewPlanner, get_default_provider
from goalreel.generation.providers.mock import MockProvider
from goalreel.generation.providers.unavailable import UnavailableProvider


def test_default_provider_unavailable(monkeypatch):
    monkeypatch.setenv("GOALREEL_NOVEL_VIEW_BACKEND", "unavailable")
    assert isinstance(get_default_provider(), UnavailableProvider)


def test_default_provider_mock(monkeypatch):
    monkeypatch.setenv("GOALREEL_NOVEL_VIEW_BACKEND", "mock")
    assert isinstance(get_default_provider(), MockProvider)


def test_mock_generate():
    p = MockProvider()
    assert p.status()["status"] == "OK"
    assert p.generate({})["status"] == "OK"


def test_unavailable_generate():
    p = UnavailableProvider()
    assert p.status()["status"] == "MODEL_UNAVAILABLE"
    assert p.generate({})["status"] == "MODEL_UNAVAILABLE"


def test_planner_status():
    planner = NovelViewPlanner(provider=UnavailableProvider())
    assert planner.status()["status"] == "MODEL_UNAVAILABLE"


def test_seva_provider_requires_backend(monkeypatch):
    monkeypatch.setenv("GOALREEL_NOVEL_VIEW_BACKEND", "seva")
    from goalreel.models.manager import ModelManager

    mgr = ModelManager()
    # virtual_camera est désactivé par défaut -> pipeline non bloqué
    inst, st = mgr.try_stage("virtual_camera")
    assert st.status in ("MODEL_UNAVAILABLE", "DISABLED")


def test_seva_provider_generate_without_weights(tmp_path):
    from goalreel.models.seva_provider import SevaProvider

    # demo.py absent -> MODEL_UNAVAILABLE honnête
    provider = SevaProvider(tmp_path, version="1.1")
    res = provider.generate({"input_dir": str(tmp_path)})
    assert res["status"] == "MODEL_UNAVAILABLE"
