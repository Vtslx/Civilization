from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import (
    Stage63RetentionConfig,
    build_stage63_fake_service,
    run_stage63_job_retention_listing_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "retention secret text",
        "memory_items": ["retention secret memory"],
        "rule_items": ["retention secret rule"],
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


def test_stage63_lists_jobs_with_pagination_and_status_filter(tmp_path) -> None:
    service = build_stage63_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_list_limit=2),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        for index in range(3):
            job_id = f"job-{index}"
            _post(f"{base}/v1/jobs", {"request": _payload(job_id)})
            wait_for_job(base, job_id)
        listed = _get(f"{base}/v1/jobs?limit=10")
        filtered = _get(f"{base}/v1/jobs?status=completed&limit=2&offset=1")
    finally:
        service.shutdown()

    assert listed["status"] == "ok"
    assert listed["jobs"]["total"] == 3
    assert len(listed["jobs"]["jobs"]) == 2
    assert filtered["jobs"]["status_filter"] == "completed"
    assert filtered["jobs"]["offset"] == 1


def test_stage63_ttl_cleanup_prunes_terminal_jobs_only(tmp_path) -> None:
    service = build_stage63_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=0.01),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(cleanup_on_persist=False),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-old")})
        wait_for_job(base, "job-old")
        time.sleep(0.03)
        cleanup = _post(f"{base}/admin/jobs/cleanup")
        listed = _get(f"{base}/v1/jobs?limit=10")
    finally:
        service.shutdown()

    assert cleanup["cleanup"]["removed"] >= 1
    assert listed["jobs"]["total"] == 0
    assert cleanup["cleanup"]["retention"]["metrics"]["ttl_pruned_jobs"] >= 1


def test_stage63_capacity_cleanup_prunes_old_terminal_jobs(tmp_path) -> None:
    state_path = tmp_path / "jobs_state.json"
    service = build_stage63_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=2),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        for index in range(3):
            job_id = f"job-capacity-{index}"
            _post(f"{base}/v1/jobs", {"request": _payload(job_id)})
            wait_for_job(base, job_id)
        admin = _get(f"{base}/admin/jobs")
        listed = _get(f"{base}/v1/jobs?limit=10")
    finally:
        service.shutdown()

    assert listed["jobs"]["total"] == 2
    assert admin["jobs"]["retention"]["metrics"]["capacity_pruned_jobs"] >= 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["jobs"]) == 2


def test_stage63_smoke_passes(tmp_path) -> None:
    summary = run_stage63_job_retention_listing_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["listing_available"]
    assert summary["stage_gates"]["capacity_pruned"]
