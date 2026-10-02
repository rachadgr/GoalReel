"""Tests du service de génération (client ComfyUI mocké)."""
from __future__ import annotations

import httpx
import pytest

from goalreel.studio.comfyui import ComfyUIClient, ComfyUIWorkflowError
from goalreel.studio.jobs import JobState
from goalreel.studio.service import GenerationRequest, GenerationService
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


def _mock_factory(handler):
    def factory():
        transport = httpx.MockTransport(handler)
        http = httpx.Client(transport=transport, base_url="http://comfy.test")
        return ComfyUIClient("http://comfy.test", client=http, max_retries=1, retry_backoff_s=0.0)

    return factory


def _ok_handler(request):
    p = request.url.path
    if p == "/upload/image":
        return httpx.Response(200, json={"name": "start.png"})
    if p == "/prompt":
        return httpx.Response(200, json={"prompt_id": "PID-42"})
    if p == "/history/PID-42":
        return httpx.Response(
            200,
            json={
                "PID-42": {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {"12": {"gifs": [{"filename": "out_00001.mp4", "type": "output"}]}},
                }
            },
        )
    if p == "/view":
        return httpx.Response(200, content=b"MP4-BYTES")
    return httpx.Response(404)


def test_full_generation_completes(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    svc = GenerationService(settings, client_factory=_mock_factory(_ok_handler))
    img = tmp_path / "in.png"
    img.write_bytes(b"PNG")
    job = svc.store.create({})
    svc.run_sync(job, GenerationRequest(image_path=str(img), preset_id="sprint"))

    d = job.to_dict()
    assert d["state"] == "COMPLETED"
    assert d["progress"] == 100
    assert d["result_url"] == f"/api/result/{job.id}"
    assert job.prompt_id == "PID-42"  # prompt_id ≠ job_id
    assert job.id != "PID-42"
    out = svc.output_path(job)
    assert out is not None and out.read_bytes() == b"MP4-BYTES"


def test_comfy_unavailable_marks_job_failed(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_dirs()

    def boom(request):
        raise httpx.ConnectError("refused")

    svc = GenerationService(settings, client_factory=_mock_factory(boom))
    img = tmp_path / "in.png"
    img.write_bytes(b"PNG")
    job = svc.store.create({})
    svc.run_sync(job, GenerationRequest(image_path=str(img)))

    assert job.state is JobState.FAILED
    assert job.error is not None
    assert "injoignable" in job.error.lower()


def test_workflow_error_marks_job_failed(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_dirs()

    def bad(request):
        p = request.url.path
        if p == "/upload/image":
            return httpx.Response(200, json={"name": "start.png"})
        if p == "/prompt":
            return httpx.Response(400, json={"error": "invalid"})
        return httpx.Response(404)

    svc = GenerationService(settings, client_factory=_mock_factory(bad))
    img = tmp_path / "in.png"
    img.write_bytes(b"PNG")
    job = svc.store.create({})
    svc.run_sync(job, GenerationRequest(image_path=str(img)))
    assert job.state is JobState.FAILED


def test_output_none_when_not_completed(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    svc = GenerationService(settings, client_factory=_mock_factory(_ok_handler))
    job = svc.store.create({})
    assert svc.output_path(job) is None


def test_start_runs_in_background(tmp_path):
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    svc = GenerationService(settings, client_factory=_mock_factory(_ok_handler))
    img = tmp_path / "in.png"
    img.write_bytes(b"PNG")
    job = svc.start(GenerationRequest(image_path=str(img), preset_id="dribble"))
    assert job.id in {j.id for j in svc.store.all()}
    # On attend la fin (courte, mock instantané).
    import time

    for _ in range(200):
        if job.state.is_terminal:
            break
        time.sleep(0.02)
    assert job.state is JobState.COMPLETED
