from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import (
    Stage66ExportConfig,
    build_stage66_fake_service,
    run_stage66_batch_export_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "export secret text",
        "memory_items": ["export secret memory"],
        "rule_items": ["export secret rule"],
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


def _service(tmp_path, *, max_export_jobs: int = 4):
    return build_stage66_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=max_export_jobs),
    )


def test_stage66_exports_completed_batch(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("job-a"), _payload("job-b")]})
        job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
        for job_id in job_ids:
            wait_for_job(base, job_id)
        exported = _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "export-a"})
        fetched = _get(f"{base}/v1/jobs/export/export-a")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    manifest = tmp_path / "exports" / "export-a" / "manifest.json"
    results = tmp_path / "exports" / "export-a" / "results.jsonl"
    failures = tmp_path / "exports" / "export-a" / "failures.jsonl"
    assert exported["status"] == "created"
    assert exported["export"]["exported_jobs"] == 2
    assert exported["export"]["exported_rows"] == 2
    assert fetched["export"]["export_id"] == "export-a"
    assert manifest.exists()
    assert results.exists()
    assert failures.exists()
    assert len(results.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert admin["jobs"]["exports"]["metrics"]["exports_created"] == 1


def test_stage66_export_records_missing_jobs_as_failures(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        exported = _post(f"{base}/v1/jobs/export", {"job_ids": ["missing"], "export_id": "export-missing"})
    finally:
        service.shutdown()

    failures = tmp_path / "exports" / "export-missing" / "failures.jsonl"
    assert exported["export"]["failures"] == 1
    assert "job_not_found" in failures.read_text(encoding="utf-8")


def test_stage66_rejects_oversized_export(tmp_path) -> None:
    service = _service(tmp_path, max_export_jobs=1)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        try:
            _post(f"{base}/v1/jobs/export", {"job_ids": ["a", "b"], "export_id": "too-large"})
            raise AssertionError("oversized export should fail")
        except HTTPError as error:
            assert error.code == 429
    finally:
        service.shutdown()


def test_stage66_smoke_passes(tmp_path) -> None:
    summary = run_stage66_batch_export_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["export_created"]
    assert summary["stage_gates"]["results_written"]
