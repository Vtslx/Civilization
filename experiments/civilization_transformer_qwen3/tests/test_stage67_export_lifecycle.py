from __future__ import annotations

import json
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import Stage66ExportConfig
from experiments.civilization_transformer_qwen3.analysis.stage67_export_lifecycle import (
    Stage67ExportLifecycleConfig,
    build_stage67_fake_service,
    run_stage67_export_lifecycle_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "lifecycle secret text",
        "memory_items": ["lifecycle secret memory"],
        "rule_items": ["lifecycle secret rule"],
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


def _service(tmp_path, *, max_persisted_exports: int = 2):
    return build_stage67_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=max_persisted_exports, max_export_list_limit=10),
    )


def _create_completed_jobs(base: str, count: int = 3) -> list[str]:
    submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload(f"job-{index}") for index in range(count)]})
    job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
    for job_id in job_ids:
        wait_for_job(base, job_id)
    return job_ids


def test_stage67_lists_exports_and_reports_integrity(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        job_ids = _create_completed_jobs(base, 3)
        _post(f"{base}/v1/jobs/export", {"job_ids": job_ids[:2], "export_id": "export-a"})
        _post(f"{base}/v1/jobs/export", {"job_ids": job_ids[1:], "export_id": "export-b"})
        listed = _get(f"{base}/v1/jobs/exports?offset=0&limit=10")
        integrity = _get(f"{base}/v1/jobs/export/export-a/integrity")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert listed["exports"]["total"] == 2
    assert {item["export_id"] for item in listed["exports"]["exports"]} == {"export-a", "export-b"}
    assert integrity["integrity"]["valid"]
    assert integrity["integrity"]["actual"]["results"]["sha256"]
    assert admin["jobs"]["exports"]["lifecycle"]["metrics"]["integrity_checks"] == 1


def test_stage67_integrity_detects_tampered_export_results(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        job_ids = _create_completed_jobs(base, 1)
        _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "export-tamper"})
        results = tmp_path / "exports" / "export-tamper" / "results.jsonl"
        results.write_text(results.read_text(encoding="utf-8") + "\n{\"tampered\": true}\n", encoding="utf-8")
        integrity = _get(f"{base}/v1/jobs/export/export-tamper/integrity")
    finally:
        service.shutdown()

    assert not integrity["integrity"]["valid"]
    assert not integrity["integrity"]["checks"]["results_sha256"]


def test_stage67_cleanup_enforces_max_persisted_exports(tmp_path) -> None:
    service = _service(tmp_path, max_persisted_exports=1)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        job_ids = _create_completed_jobs(base, 3)
        _post(f"{base}/v1/jobs/export", {"job_ids": job_ids[:1], "export_id": "export-a"})
        _post(f"{base}/v1/jobs/export", {"job_ids": job_ids[1:2], "export_id": "export-b"})
        cleanup = _post(f"{base}/v1/jobs/exports/cleanup", {"ttl_seconds": 3600, "max_exports": 1})
        listed = _get(f"{base}/v1/jobs/exports?offset=0&limit=10")
    finally:
        service.shutdown()

    assert cleanup["cleanup"]["deleted"] == 1
    assert listed["exports"]["total"] == 1


def test_stage67_smoke_passes(tmp_path) -> None:
    summary = run_stage67_export_lifecycle_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["integrity_valid"]
    assert summary["stage_gates"]["cleanup_deleted"]
