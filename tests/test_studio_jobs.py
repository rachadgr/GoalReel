"""Tests du cycle de vie et du store de jobs."""
from __future__ import annotations

from goalreel.studio.jobs import Job, JobState, JobStore


def test_job_initial_state():
    job = Job(id="abc")
    assert job.state is JobState.QUEUED
    assert job.progress == 0
    assert job.result_url is None and job.error is None


def test_job_update_and_timestamps():
    job = Job(id="abc")
    created = job.created_at
    job.update(state=JobState.GENERATING, progress=50, message="go")
    assert job.state is JobState.GENERATING
    assert job.progress == 50
    assert job.message == "go"
    assert job.created_at == created
    assert job.updated_at >= created


def test_progress_is_clamped():
    job = Job(id="abc")
    job.update(progress=150)
    assert job.progress == 100
    job.update(progress=-20)
    assert job.progress == 0


def test_to_dict_schema_is_stable():
    job = Job(id="abc")
    job.update(state=JobState.COMPLETED, progress=100, result_url="/api/result/abc")
    d = job.to_dict()
    assert set(d.keys()) == {
        "job_id",
        "state",
        "progress",
        "message",
        "created_at",
        "updated_at",
        "result_url",
        "error",
        "request",
    }
    assert d["job_id"] == "abc"
    assert d["state"] == "COMPLETED"


def test_terminal_states():
    assert JobState.COMPLETED.is_terminal
    assert JobState.FAILED.is_terminal
    assert not JobState.GENERATING.is_terminal


def test_store_unique_ids():
    store = JobStore()
    ids = {store.create().id for _ in range(50)}
    assert len(ids) == 50


def test_store_get_and_recent():
    store = JobStore()
    j1 = store.create({"a": 1})
    j2 = store.create({"b": 2})
    assert store.get(j1.id) is j1
    assert store.get("missing") is None
    recent = store.recent(10)
    assert recent[0].id == j2.id  # plus récent en premier


def test_store_eviction_fifo():
    store = JobStore(max_jobs=3)
    ids = [store.create().id for _ in range(5)]
    assert store.count() == 3
    assert store.get(ids[0]) is None  # évincé
    assert store.get(ids[-1]) is not None


def test_prompt_id_separate_from_job_id():
    store = JobStore()
    job = store.create()
    job.update(prompt_id="comfy-999")
    assert job.id != job.prompt_id
    assert job.prompt_id == "comfy-999"
