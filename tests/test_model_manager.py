"""Tests du ModelManager : chargement, cache, unload, robustesse."""
import pytest

from goalreel.models.manager import ModelManager, ModelUnavailable


class Dummy:
    def __init__(self, tag="d"):
        self.tag = tag


def test_register_and_load(manager):
    manager.register("dummy", lambda spec, mgr: Dummy())
    manager.settings.models["dummy"] = type(manager.spec("detector"))(
        name="dummy", stage="test", path=None)
    inst = manager.load("dummy")
    assert isinstance(inst, Dummy)
    # cache : même instance au 2e appel
    assert manager.load("dummy") is inst


def test_missing_checkpoint_unavailable(manager):
    # detector configuré vers un chemin inexistant
    manager.spec("detector").path = "/nonexistent/weights.pt"
    inst, st = manager.try_stage("detector")
    assert inst is None
    assert st.status == "MODEL_UNAVAILABLE"


def test_disabled_model(manager):
    manager.spec("depth").enabled = False
    inst, st = manager.try_stage("depth")
    assert inst is None
    assert st.status in ("MODEL_UNAVAILABLE", "DISABLED")


def test_loader_unavailable_runtime(manager):
    def bad_loader(spec, mgr):
        raise ModelUnavailable("runtime missing")

    manager.register("depth", bad_loader)
    inst, st = manager.try_stage("depth")
    assert inst is None
    assert st.status == "MODEL_UNAVAILABLE"


def test_loader_real_error_propagates_on_load(manager):
    def boom(spec, mgr):
        raise ValueError("boom")

    manager.register("depth", boom)
    with pytest.raises(ValueError):
        manager.load("depth")


def test_unload(manager):
    manager.register("dummy", lambda spec, mgr: Dummy())
    from goalreel.config import ModelSpec
    manager.settings.models["dummy"] = ModelSpec(name="dummy", stage="test", path=None)
    inst = manager.load("dummy")
    assert inst is not None
    manager.unload("dummy")
    assert manager.state("dummy").loaded is False


def test_inference_timing(manager):
    manager.register("dummy", lambda spec, mgr: Dummy())
    from goalreel.config import ModelSpec
    manager.settings.models["dummy"] = ModelSpec(name="dummy", stage="test", path=None)
    manager.load("dummy")
    with manager.time_it("dummy"):
        pass
    st = manager.state("dummy")
    assert st.infer_count == 1
    assert st.infer_time_s >= 0


def test_statuses_shape(manager):
    st = manager.statuses()
    assert "detector" in st
    for v in st.values():
        for key in ("name", "stage", "status", "device", "loaded"):
            assert key in v
