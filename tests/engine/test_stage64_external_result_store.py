from __future__ import annotations

import json
from urllib.request import Request, urlopen

from civilization.engine.stages.stage61_async_jobs import Stage61JobConfig, wait_for_job
from civilization.engine.stages.stage62_persistent_async_jobs import Stage62PersistenceConfig
from civilization.engine.stages.stage63_job_retention_listing import Stage63RetentionConfig
from civilization.engine.stages.stage64_external_result_store import (
    Stage64ResultStoreConfig,
    build_stage64_fake_service,
    run_stage64_external_result_store_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "external result secret text",
        "memory_items": ["external result secret memory"],
        "rule_items": ["external result secret rule"],
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


def test_stage64_externalizes_result_rows_and_pages_them(tmp_path) -> None:
    state = tmp_path / "jobs_state.json"
    results = tmp_path / "results"
    service = build_stage64_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=10),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(results), inline_result_row_limit=0, max_result_page_limit=1),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-external")})
        completed = wait_for_job(base, "job-external")
        summary = _get(f"{base}/v1/jobs/job-external")
        page = _get(f"{base}/v1/jobs/job-external/result?limit=5")
    finally:
        service.shutdown()

    assert completed["job"]["status"] == "completed"
    assert any(results.glob("*.jsonl"))
    state_text = state.read_text(encoding="utf-8")
    assert '"rows":' not in state_text
    assert summary["job"]["result"]["result_ref"]["storage"] == "jsonl"
    assert "rows" not in summary["job"]["result"]
    assert page["result"]["result_available"]
    assert page["result"]["total_rows"] >= 1
    assert len(page["result"]["rows"]) == 1


def test_stage64_restores_result_ref_after_restart(tmp_path) -> None:
    state = tmp_path / "jobs_state.json"
    results = tmp_path / "results"
    job_config = Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=3600)
    persistence = Stage62PersistenceConfig(job_state_path=str(state))
    retention = Stage63RetentionConfig(max_persisted_jobs=10)
    result_store = Stage64ResultStoreConfig(result_dir=str(results), inline_result_row_limit=0)

    first = build_stage64_fake_service(
        port=0,
        job_config=job_config,
        persistence_config=persistence,
        retention_config=retention,
        result_store_config=result_store,
    )
    server = first.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-restart")})
        wait_for_job(base, "job-restart")
    finally:
        first.shutdown()

    second = build_stage64_fake_service(
        port=0,
        job_config=job_config,
        persistence_config=persistence,
        retention_config=retention,
        result_store_config=result_store,
    )
    server = second.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        summary = _get(f"{base}/v1/jobs/job-restart")
        page = _get(f"{base}/v1/jobs/job-restart/result?limit=1")
        admin = _get(f"{base}/admin/jobs")
    finally:
        second.shutdown()

    assert summary["job"]["status"] == "completed"
    assert summary["job"]["result"]["result_ref"]
    assert page["result"]["result_available"]
    assert page["result"]["rows"]
    assert "result_store" in admin["jobs"]


def test_stage64_prune_deletes_external_result_file(tmp_path) -> None:
    state = tmp_path / "jobs_state.json"
    results = tmp_path / "results"
    service = build_stage64_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=1),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(results), inline_result_row_limit=0),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post(f"{base}/v1/jobs", {"request": _payload("job-old")})
        wait_for_job(base, "job-old")
        first_files = set(results.glob("*.jsonl"))
        _post(f"{base}/v1/jobs", {"request": _payload("job-new")})
        wait_for_job(base, "job-new")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    remaining = set(results.glob("*.jsonl"))
    assert first_files
    assert len(remaining) == 1
    assert admin["jobs"]["result_store"]["metrics"]["result_files_deleted"] >= 1


def test_stage64_smoke_passes(tmp_path) -> None:
    summary = run_stage64_external_result_store_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["result_file_written"]
    assert summary["stage_gates"]["state_does_not_inline_rows"]
