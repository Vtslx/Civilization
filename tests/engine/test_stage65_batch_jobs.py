from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from civilization.engine.stages.stage61_async_jobs import Stage61JobConfig, wait_for_job
from civilization.engine.stages.stage62_persistent_async_jobs import Stage62PersistenceConfig
from civilization.engine.stages.stage63_job_retention_listing import Stage63RetentionConfig
from civilization.engine.stages.stage64_external_result_store import Stage64ResultStoreConfig
from civilization.engine.stages.stage65_batch_jobs import (
    Stage65BatchConfig,
    build_stage65_fake_service,
    run_stage65_batch_jobs_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "batch secret text",
        "memory_items": ["batch secret memory"],
        "rule_items": ["batch secret rule"],
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


def _service(tmp_path, *, max_batch_submit: int = 4, max_batch_result_jobs: int = 4):
    return build_stage65_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=max_batch_submit, max_batch_result_jobs=max_batch_result_jobs),
    )


def test_stage65_batch_submit_and_batch_results(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"job-batch-{index}") for index in range(3)]
        submitted = _post(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        batch_results = _post(f"{base}/v1/jobs/results", {"job_ids": job_ids, "limit": 5})
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert submitted["status"] == "accepted"
    assert submitted["batch"]["accepted_count"] == 3
    assert all(item["job"]["status"] == "completed" for item in completed)
    assert len(batch_results["batch_results"]["results"]) == 3
    assert all(len(item["rows"]) == 1 for item in batch_results["batch_results"]["results"])
    assert admin["jobs"]["batch"]["metrics"]["batch_jobs_accepted"] == 3


def test_stage65_batch_submit_rejects_oversized_batch(tmp_path) -> None:
    service = _service(tmp_path, max_batch_submit=2)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        try:
            _post(f"{base}/v1/jobs/batch", {"requests": [_payload("a"), _payload("b"), _payload("c")]})
            raise AssertionError("oversized batch should fail")
        except HTTPError as error:
            assert error.code == 429
    finally:
        service.shutdown()


def test_stage65_batch_results_rejects_oversized_job_list(tmp_path) -> None:
    service = _service(tmp_path, max_batch_result_jobs=1)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        try:
            _post(f"{base}/v1/jobs/results", {"job_ids": ["a", "b"], "limit": 1})
            raise AssertionError("oversized result batch should fail")
        except HTTPError as error:
            assert error.code == 429
    finally:
        service.shutdown()


def test_stage65_partial_batch_reports_invalid_items(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        payload = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("valid"), "bad"]})
        assert payload["status"] == "partial"
        assert payload["batch"]["accepted_count"] == 1
        assert payload["batch"]["rejected_count"] == 1
    finally:
        service.shutdown()


def test_stage65_smoke_passes(tmp_path) -> None:
    summary = run_stage65_batch_jobs_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["batch_submit_accepted"]
    assert summary["stage_gates"]["batch_results_available"]
