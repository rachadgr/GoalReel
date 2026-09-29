"""Fixtures partagées pour les tests GoalReel."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture
def synthetic_frame():
    """Frame BGR synthétique (vert 'terrain' + formes)."""
    import cv2

    frame = np.zeros((576, 1024, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 40)
    cv2.circle(frame, (300, 300), 40, (230, 230, 230), -1)
    cv2.rectangle(frame, (600, 250), (680, 500), (20, 20, 200), -1)
    return frame


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from goalreel.config import load_settings

    monkeypatch.setenv("GOALREEL_MODELS_DIR", str(ROOT / "models"))
    monkeypatch.setenv("GOALREEL_CHECKPOINTS_DIR", str(ROOT / "checkpoints"))
    monkeypatch.setenv("GOALREEL_VENDOR_DIR", str(ROOT / "vendor"))
    monkeypatch.setenv("GOALREEL_CACHE_DIR", str(tmp_path / "cache"))
    return load_settings()


@pytest.fixture
def manager(settings):
    from goalreel.models.manager import ModelManager

    return ModelManager(settings=settings)
