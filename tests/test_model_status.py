"""Tests des *états véridiques* du ModelManager.

Vérifie que chaque situation réelle produit l'état honnête correspondant, sans
jamais présenter un faux succès (READY seulement si réellement chargé), tout en
conservant la rétro-compatibilité des états pipeline.
"""
from pathlib import Path

import pytest

from goalreel.config import ModelSpec, load_settings
from goalreel.models.manager import ModelManager, ModelUnavailable
from goalreel.models.status import pipeline_status


class _Dummy:
    pass


def _register_dummy(manager, name="dummy"):
    manager.register(name, lambda spec, mgr: _Dummy())
    manager.settings.models[name] = ModelSpec(name=name, stage="test", path=None)


def test_pipeline_status_mapping():
    assert pipeline_status("READY") == "OK"
    assert pipeline_status("CHECKPOINT_MISSING") == "MODEL_UNAVAILABLE"
    assert pipeline_status("DEPENDENCY_MISSING") == "MODEL_UNAVAILABLE"
    assert pipeline_status("INCOMPATIBLE") == "MODEL_UNAVAILABLE"
    assert pipeline_status("GPU_REQUIRED") == "MODEL_UNAVAILABLE"
    assert pipeline_status("LOAD_ERROR") == "ERROR"
    assert pipeline_status("DISABLED") == "MODEL_UNAVAILABLE"


def test_missing_checkpoint_truth(manager):
    manager.spec("detector").path = "/nonexistent/weights.pt"
    inst, st = manager.try_stage("detector")
    assert inst is None
    # Rétro-compat : l'état pipeline reste MODEL_UNAVAILABLE ...
    assert st.status == "MODEL_UNAVAILABLE"
    # ... et le détail honnête est exposé.
    assert st.metrics.get("truth") == "CHECKPOINT_MISSING"
    assert manager.state("detector").truth == "CHECKPOINT_MISSING"


def test_dependency_missing_truth(manager):
    def loader(spec, mgr):
        raise ModelUnavailable("runtime absent", truth="DEPENDENCY_MISSING")

    _register_dummy(manager)
    manager.register("dummy", loader)
    inst, st = manager.try_stage("dummy")
    assert inst is None
    assert st.status == "MODEL_UNAVAILABLE"
    assert manager.state("dummy").truth == "DEPENDENCY_MISSING"


def test_incompatible_truth(manager):
    def loader(spec, mgr):
        raise ModelUnavailable("arch mismatch", truth="INCOMPATIBLE")

    _register_dummy(manager)
    manager.register("dummy", loader)
    manager.try_stage("dummy")
    assert manager.state("dummy").truth == "INCOMPATIBLE"
    assert manager.state("dummy").status == "MODEL_UNAVAILABLE"


def test_gpu_required_truth(manager):
    def loader(spec, mgr):
        raise ModelUnavailable("needs cuda", truth="GPU_REQUIRED")

    _register_dummy(manager)
    manager.register("dummy", loader)
    manager.try_stage("dummy")
    assert manager.state("dummy").truth == "GPU_REQUIRED"


def test_load_error_is_error_status(manager):
    def boom(spec, mgr):
        raise ValueError("boom")

    manager.spec("depth").path = None  # on saute le contrôle de checkpoint
    manager.register("depth", boom)
    with pytest.raises(ValueError):
        manager.load("depth")
    st = manager.state("depth")
    assert st.status == "ERROR"
    assert st.truth == "LOAD_ERROR"


def test_ready_only_when_loaded(manager):
    _register_dummy(manager)
    assert manager.state("dummy").truth != "READY"
    manager.load("dummy")
    st = manager.state("dummy")
    assert st.loaded is True
    assert st.truth == "READY"
    assert st.status == "OK"


def test_disabled_truth(manager):
    manager.spec("depth").enabled = False
    manager.load("depth")
    assert manager.state("depth").truth == "DISABLED"


def test_diagnose_and_report(manager):
    _register_dummy(manager)
    diag = manager.diagnose("dummy")
    assert diag["name"] == "dummy"
    assert diag["truth"] == "READY"
    assert diag["loaded"] is True
    report = manager.readiness_report()
    assert "detector" in report and "truth" in report["detector"]


def test_statuses_include_truth(manager):
    st = manager.statuses()
    for v in st.values():
        assert "truth" in v
