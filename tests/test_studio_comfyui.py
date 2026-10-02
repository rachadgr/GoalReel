"""Tests du client ComfyUI (mocké, sans GPU ni serveur réel)."""
from __future__ import annotations

import httpx
import pytest

from goalreel.studio.comfyui import (
    ComfyUIClient,
    ComfyUIError,
    ComfyUIUnavailable,
    ComfyUIWorkflowError,
    OutputRef,
    extract_output,
)


def _client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://test")
    return ComfyUIClient(
        "http://test", client=http, max_retries=2, retry_backoff_s=0.0, **kwargs
    )


def test_status_available():
    c = _client(lambda r: httpx.Response(200, json={"system": {"v": "1"}}))
    st = c.status()
    assert st.available is True
    assert st.to_dict()["available"] is True


def test_status_unavailable_on_connection_error():
    def boom(request):
        raise httpx.ConnectError("refused")

    c = _client(boom)
    st = c.status()
    assert st.available is False
    assert "injoignable" in st.message.lower()


def test_upload_image_success(tmp_path):
    img = tmp_path / "start.png"
    img.write_bytes(b"\x89PNG\r\n")

    def handler(request):
        assert request.url.path == "/upload/image"
        return httpx.Response(200, json={"name": "start.png", "type": "input"})

    c = _client(handler)
    out = c.upload_image(img)
    assert out["name"] == "start.png"


def test_upload_missing_file_raises(tmp_path):
    c = _client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ComfyUIError):
        c.upload_image(tmp_path / "nope.png")


def test_submit_returns_prompt_id():
    c = _client(lambda r: httpx.Response(200, json={"prompt_id": "PID-1"}))
    pid = c.submit({"1": {"class_type": "X", "inputs": {}}})
    assert pid == "PID-1"


def test_submit_400_raises_workflow_error():
    c = _client(lambda r: httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(ComfyUIWorkflowError):
        c.submit({})


def test_submit_without_prompt_id_raises():
    c = _client(lambda r: httpx.Response(200, json={"node_errors": {"1": "x"}}))
    with pytest.raises(ComfyUIWorkflowError):
        c.submit({})


def test_poll_returns_entry_when_completed():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "PID": {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {"12": {"gifs": [{"filename": "o.mp4", "type": "output"}]}},
                }
            },
        )

    c = _client(handler)
    entry = c.poll("PID", timeout_s=2, interval_s=0.001)
    assert entry["status"]["completed"] is True


def test_poll_raises_on_error_status():
    def handler(request):
        return httpx.Response(
            200, json={"PID": {"status": {"status_str": "error", "completed": True}}}
        )

    c = _client(handler)
    with pytest.raises(ComfyUIWorkflowError):
        c.poll("PID", timeout_s=2, interval_s=0.001)


def test_poll_times_out():
    c = _client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ComfyUIWorkflowError):
        c.poll("PID", timeout_s=0.05, interval_s=0.01)


def test_view_downloads_bytes():
    c = _client(lambda r: httpx.Response(200, content=b"MP4"))
    assert c.view("o.mp4") == b"MP4"


def test_extract_output_prefers_video():
    entry = {
        "outputs": {
            "12": {"gifs": [{"filename": "clip.mp4", "subfolder": "s", "type": "output"}]},
            "11": {"images": [{"filename": "frame.png", "type": "output"}]},
        }
    }
    ref = extract_output(entry)
    assert ref.filename == "clip.mp4"
    assert ref.kind == "video"
    assert ref.subfolder == "s"


def test_extract_output_falls_back_to_image():
    entry = {"outputs": {"11": {"images": [{"filename": "frame.png", "type": "output"}]}}}
    ref = extract_output(entry)
    assert ref.kind == "image"


def test_extract_output_raises_when_empty():
    with pytest.raises(ComfyUIWorkflowError):
        extract_output({"outputs": {}})


def test_output_ref_to_dict():
    ref = OutputRef(filename="a.mp4", subfolder="x", folder_type="output", kind="video")
    assert ref.to_dict()["filename"] == "a.mp4"


def test_unavailable_after_retries():
    def boom(request):
        raise httpx.ConnectError("refused")

    c = _client(boom)
    with pytest.raises(ComfyUIUnavailable):
        c.system_stats()
