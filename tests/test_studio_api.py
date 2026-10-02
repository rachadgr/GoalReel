"""Tests de l'API FastAPI du Reel Studio (TestClient)."""
from __future__ import annotations

import io

import httpx
import pytest

from goalreel.studio.api import create_app, sanitize_filename, _validate_image_bytes
from goalreel.studio.comfyui import ComfyUIClient
from goalreel.studio.studio_config import StudioSettings


def _settings(tmp_path) -> StudioSettings:
    return StudioSettings(
        comfy_url="http://comfy.test",
        host="0.0.0.0",
        port=7860,
        max_upload_mb=20,
        cors_origins=["*"],
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "reels",
        job_timeout_s=5,
        poll_interval_s=0.001,
        max_retries=1,
        retry_backoff_s=0.0,
        http_timeout_s=5.0,
    )


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48, 48), (20, 180, 20)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    app = create_app(_settings(tmp_path))
    return TestClient(app)


# ---------------------------------------------------------------- helpers
def test_sanitize_filename_strips_paths():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my file (1).png") == "my_file_1_.png"
    assert sanitize_filename("") == "upload"


def test_validate_image_bytes_rejects_non_image():
    with pytest.raises(ValueError):
        _validate_image_bytes(b"not an image", "image/png")


def test_validate_image_bytes_rejects_bad_mime():
    with pytest.raises(ValueError):
        _validate_image_bytes(_png_bytes(), "text/plain")


# ---------------------------------------------------------------- endpoints
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_compat_alias(client):
    assert client.get("/health").json()["status"] == "ok"


def test_presets_endpoint(client):
    r = client.get("/api/presets")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "cinematic_football"
    assert len(body["presets"]) == 8


def test_config_endpoint_no_secrets(client):
    body = client.get("/api/config").json()
    assert "comfy_url" in body
    assert "token" not in str(body).lower()


def test_status_endpoint_reports_comfy_offline(client):
    body = client.get("/api/status").json()
    assert body["comfy"]["available"] is False


def test_upload_valid_image(client):
    r = client.post(
        "/api/upload",
        files={"image": ("start.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["upload_id"]
    assert body["filename"].endswith(".png")


def test_upload_rejects_bad_content(client):
    r = client.post(
        "/api/upload",
        files={"image": ("x.png", b"garbage", "image/png")},
    )
    assert r.status_code == 400


def test_upload_rejects_oversized(tmp_path):
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    # Limite très basse pour forcer le 413.
    object.__setattr__(settings, "max_upload_mb", 0)
    app = create_app(settings)
    c = TestClient(app)
    r = c.post("/api/upload", files={"image": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 413


def test_generate_requires_image(client):
    r = client.post("/api/generate", data={"preset": "sprint"})
    assert r.status_code == 400


def test_generate_unknown_image_404(client):
    r = client.post("/api/generate", data={"upload_id": "deadbeef", "preset": "sprint"})
    assert r.status_code == 404


def test_generate_and_poll_with_mock_comfy(tmp_path, monkeypatch):
    """Bout-en-bout via API : upload → generate → poll → result (ComfyUI mocké)."""
    from fastapi.testclient import TestClient
    from goalreel.studio import service as service_mod

    def _ok_handler(request):
        p = request.url.path
        if p == "/upload/image":
            return httpx.Response(200, json={"name": "start.png"})
        if p == "/prompt":
            return httpx.Response(200, json={"prompt_id": "P-1"})
        if p == "/history/P-1":
            return httpx.Response(
                200,
                json={
                    "P-1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {"12": {"gifs": [{"filename": "o.mp4", "type": "output"}]}},
                    }
                },
            )
        if p == "/view":
            return httpx.Response(200, content=b"MP4DATA")
        return httpx.Response(404)

    def factory():
        http = httpx.Client(transport=httpx.MockTransport(_ok_handler), base_url="http://comfy.test")
        return ComfyUIClient("http://comfy.test", client=http, max_retries=1, retry_backoff_s=0.0)

    settings = _settings(tmp_path)
    app = create_app(settings)
    # Injecte la fabrique mockée dans le service partagé.
    app.state.service._client_factory = factory
    c = TestClient(app)

    up = c.post("/api/upload", files={"image": ("s.png", _png_bytes(), "image/png")}).json()
    gen = c.post("/api/generate", data={"upload_id": up["upload_id"], "preset": "shot_on_goal"})
    assert gen.status_code == 200
    job_id = gen.json()["job_id"]

    # Attente de la fin du thread de fond.
    import time

    state = None
    for _ in range(200):
        body = c.get(f"/api/status/{job_id}").json()
        state = body["state"]
        if state in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.02)
    assert state == "COMPLETED"

    res = c.get(f"/api/result/{job_id}")
    assert res.status_code == 200
    assert res.content == b"MP4DATA"


def test_result_conflict_when_not_completed(client):
    # Job inexistant → 404
    assert client.get("/api/result/nope").status_code == 404


def test_job_status_404(client):
    assert client.get("/api/status/nope").status_code == 404


def test_jobs_listing(client):
    body = client.get("/api/jobs").json()
    assert "jobs" in body


def test_generate_compat_alias_requires_image(client):
    r = client.post("/generate", data={"preset": "sprint"})
    assert r.status_code == 400
