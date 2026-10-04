from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen

from civilization.engine.stages.stage61_async_jobs import Stage61JobConfig, wait_for_job
from civilization.engine.stages.stage62_persistent_async_jobs import (
    Stage62PersistenceConfig,
    build_stage62_fake_service,
    run_stage62_persistent_async_jobs_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "persistent secret text",
        "memory_items": ["persistent secret memory"],
        "rule_items": ["persistent secret rule"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def _post(url: str, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def _delete(url: str) -> dict:
    request = Request(url, method="DELETE")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def test_stage62_restores_completed_and_canceled_jobs(tmp_path) -> None:
    state = tmp_path / "jobs_state.json"
    log = tmp_path / "jobs.jsonl"
    job_config = Stage61JobConfig(job_log_path=str(log), max_queued_jobs=4)
    persistence = Stage62PersistenceConfig(job_state_path=str(state))

    first = build_stage62_fake_service(port=0, job_config=job_config, persistence_config=persistence, runtime_delay_seconds=0.05)
    server = first.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-completed")})
        completed = wait_for_job(base, "job-completed")
        submitted = _post(f"{base}/v1/jobs", {"request": _payload("job-canceled")})
        canceled = _delete(f"{base}/v1/jobs/{submitted['job']['job_id']}")
    finally:
        first.shutdown()

    assert completed["job"]["status"] == "completed"
    assert canceled["job"]["status"] in {"queued", "running", "canceled", "completed"}
    assert state.exists()

    second = build_stage62_fake_service(port=0, job_config=job_config, persistence_config=persistence)
    server = second.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        restored_completed = _get(f"{base}/v1/jobs/job-completed")
        restored_canceled = _get(f"{base}/v1/jobs/job-canceled")
        admin = _get(f"{base}/admin/jobs")
    finally:
        second.shutdown()

    assert restored_completed["job"]["status"] == "completed"
    assert restored_canceled["job"]["status"] in {"canceled", "completed"}
    assert admin["jobs"]["persistence"]["restored_job_count"] >= 2
    assert admin["jobs"]["persistence"]["state_file_exists"]


def test_stage62_recovers_queued_job_from_state_file(tmp_path) -> None:
    created_at = time.time()
    state = tmp_path / "jobs_state.json"
    state.write_text(
        json.dumps(
            {
                "stage": "stage62_persistent_async_jobs",
                "updated_at": created_at,
                "jobs": [
                    {
                        "job_id": "job-recovered",
                        "status": "queued",
                        "created_at": created_at,
                        "updated_at": created_at,
                        "payload": _payload("job-recovered"),
                        "result": None,
                        "error": None,
                        "cancel_requested": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    job_config = Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=4)
    persistence = Stage62PersistenceConfig(job_state_path=str(state))
    service = build_stage62_fake_service(port=0, job_config=job_config, persistence_config=persistence)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        completed = wait_for_job(base, "job-recovered")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert completed["job"]["status"] == "completed"
    assert admin["jobs"]["persistence"]["requeued_job_count"] == 1


def test_stage62_persistence_log_is_redacted(tmp_path) -> None:
    log = tmp_path / "jobs.jsonl"
    state = tmp_path / "jobs_state.json"
    service = build_stage62_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(log)),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state)),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-redacted")})
        wait_for_job(base, "job-redacted")
    finally:
        service.shutdown()

    text = log.read_text(encoding="utf-8")
    assert "persistent secret text" not in text
    assert "persistent secret memory" not in text
    assert "persistent secret rule" not in text
    assert "job_submitted" in text
    assert "job_completed" in text


def test_stage62_smoke_passes(tmp_path) -> None:
    summary = run_stage62_persistent_async_jobs_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["completed_restored"]
    assert summary["stage_gates"]["admin_persistence_visible"]
