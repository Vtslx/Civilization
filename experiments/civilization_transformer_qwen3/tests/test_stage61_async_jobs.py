from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import (
    Stage61JobConfig,
    build_stage61_fake_service,
    run_stage61_async_jobs_smoke,
    wait_for_job,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "async secret text",
        "memory_items": ["secret memory"],
        "rule_items": ["secret rule"],
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


def test_stage61_submit_and_complete_job(tmp_path) -> None:
    service = build_stage61_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl")),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        submitted = _post(f"{base}/v1/jobs", {"request": _payload("job-complete")})
        completed = wait_for_job(base, "job-complete")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()
    assert submitted["status"] == "accepted"
    assert completed["job"]["status"] == "completed"
    assert completed["job"]["result"]["status"] == "ok"
    assert admin["jobs"]["metrics"]["completed"] >= 1


def test_stage61_cancel_queued_job(tmp_path) -> None:
    service = build_stage61_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=4),
        runtime_delay_seconds=0.3,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-running")})
        submitted = _post(f"{base}/v1/jobs", {"request": _payload("job-cancel")})
        canceled = _delete(f"{base}/v1/jobs/{submitted['job']['job_id']}")
        fetched = _get(f"{base}/v1/jobs/{submitted['job']['job_id']}")
    finally:
        service.shutdown()
    assert canceled["status"] == "ok"
    assert fetched["job"]["status"] in {"canceled", "running", "completed"}


def test_stage61_queue_full_rejects_jobs(tmp_path) -> None:
    service = build_stage61_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=1),
        runtime_delay_seconds=0.5,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-1")})
        _post(f"{base}/v1/jobs", {"request": _payload("job-2")})
        try:
            _post(f"{base}/v1/jobs", {"request": _payload("job-3")})
            raise AssertionError("third queued job should be rejected")
        except HTTPError as error:
            assert error.code == 429
    finally:
        service.shutdown()


def test_stage61_job_log_is_redacted(tmp_path) -> None:
    log = tmp_path / "jobs.jsonl"
    service = build_stage61_fake_service(port=0, job_config=Stage61JobConfig(job_log_path=str(log)))
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-redacted")})
        wait_for_job(base, "job-redacted")
    finally:
        service.shutdown()
    text = log.read_text(encoding="utf-8")
    assert "async secret text" not in text
    assert "secret memory" not in text
    assert "job_submitted" in text
    assert "job_completed" in text


def test_stage61_smoke_passes(tmp_path) -> None:
    summary = run_stage61_async_jobs_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["job_completed"]
    assert summary["stage_gates"]["job_log_written"]
