from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import Stage66ExportConfig
from experiments.civilization_transformer_qwen3.analysis.stage67_export_lifecycle import Stage67ExportLifecycleConfig
from experiments.civilization_transformer_qwen3.analysis.stage68_export_package_delivery import Stage68PackageConfig
from experiments.civilization_transformer_qwen3.analysis.stage69_streaming_package_delivery import Stage69StreamingConfig
from experiments.civilization_transformer_qwen3.analysis.stage70_range_package_delivery import (
    build_stage70_fake_service,
    run_stage70_range_package_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "range package secret text",
        "memory_items": ["range package secret memory"],
        "rule_items": ["range package secret rule"],
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


def _download(url: str, *, range_header: str | None = None) -> tuple[int, dict[str, str], bytes]:
    headers = {"Range": range_header} if range_header else {}
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return response.status, dict(response.headers), response.read()


def _service(tmp_path):
    return build_stage70_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
        streaming_config=Stage69StreamingConfig(download_chunk_bytes=32),
    )


def _create_package(base: str, export_id: str = "export-range") -> None:
    submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("job-a"), _payload("job-b"), _payload("job-c")]})
    job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
    for job_id in job_ids:
        wait_for_job(base, job_id)
    _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": export_id})
    _post(f"{base}/v1/jobs/export/{export_id}/package", {})


def test_stage70_supports_prefix_and_suffix_ranges(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_package(base)
        status_full, headers_full, full = _download(f"{base}/v1/jobs/export/export-range/download")
        status_prefix, headers_prefix, prefix = _download(f"{base}/v1/jobs/export/export-range/download", range_header="bytes=0-31")
        status_suffix, headers_suffix, suffix = _download(f"{base}/v1/jobs/export/export-range/download", range_header="bytes=-16")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert status_full == 200
    assert headers_full["X-Download-Mode"] == "streaming"
    assert status_prefix == 206
    assert headers_prefix["X-Download-Mode"] == "range-streaming"
    assert headers_prefix["Content-Range"] == f"bytes 0-31/{len(full)}"
    assert prefix == full[:32]
    assert status_suffix == 206
    assert suffix == full[-16:]
    assert headers_suffix["Content-Range"] == f"bytes {len(full) - 16}-{len(full) - 1}/{len(full)}"
    metrics = admin["jobs"]["exports"]["range_delivery"]["metrics"]
    assert metrics["range_requests"] == 2
    assert metrics["partial_downloads"] == 2
    assert metrics["ranged_bytes"] == 48


def test_stage70_rejects_unsatisfiable_range(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_package(base)
        size = _get(f"{base}/v1/jobs/export/export-range/package")["package"]["size_bytes"]
        try:
            _download(f"{base}/v1/jobs/export/export-range/download", range_header=f"bytes={size + 10}-{size + 20}")
            raise AssertionError("unsatisfiable range should fail")
        except HTTPError as error:
            assert error.code == 416
            assert error.headers["Content-Range"] == f"bytes */{size}"
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert admin["jobs"]["exports"]["range_delivery"]["metrics"]["rejected_ranges"] == 1


def test_stage70_smoke_passes(tmp_path) -> None:
    summary = run_stage70_range_package_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["range_body_size"]
    assert summary["stage_gates"]["range_metrics_visible"]
