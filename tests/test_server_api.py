"""Tests de l'API serveur (endpoints modèles / pipeline)."""
import json
import threading
import urllib.request
from http.server import HTTPServer

import pytest


@pytest.fixture(scope="module")
def api_base():
    import server
    httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read().decode())


def test_health(api_base):
    assert _get(api_base, "/health")["status"] == "ok"


def test_models_endpoint(api_base):
    data = _get(api_base, "/models")
    assert "models" in data and isinstance(data["models"], list)
    names = {m["name"] for m in data["models"]}
    assert {"detector", "segmenter", "depth", "reid"}.issubset(names)


def test_pipeline_endpoint(api_base):
    data = _get(api_base, "/pipeline")
    assert any(s["id"] == "final_reel" for s in data["stages"])


def test_pipeline_status(api_base):
    data = _get(api_base, "/pipeline/status")
    assert len(data["stages"]) > 10
    for s in data["stages"]:
        assert "status" in s


def test_unknown_model_404(api_base):
    with pytest.raises(urllib.error.HTTPError):
        _get(api_base, "/models/does_not_exist")
