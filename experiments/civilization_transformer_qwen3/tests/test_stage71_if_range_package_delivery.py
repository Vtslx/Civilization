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
from experiments.civilization_transformer_qwen3.analysis.stage71_if_range_package_delivery import (
    build_stage71_fake_service,
    run_stage71_if_range_package_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "if-range package secret text",
        "memory_items": ["if-range package secret memory"],
        "rule_items": ["if-range package secret rule"],
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


def _download(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, headers=headers or {}, method="GET")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return response.status, dict(response.headers), response.read()


def _service(tmp_path):
    return build_stage71_fake_service(
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


def _create_package(base: str, export_id: str = "export-if-range") -> None:
    submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("job-a"), _payload("job-b"), _payload("job-c")]})
    job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
    for job_id in job_ids:
        wait_for_job(base, job_id)
    _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": export_id})
    _post(f"{base}/v1/jobs/export/{export_id}/package", {})


def test_stage71_etag_and_if_range_match(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_package(base)
        status_full, headers_full, full = _download(f"{base}/v1/jobs/export/export-if-range/download")
        etag = headers_full["ETag"]
        status_partial, headers_partial, partial = _download(
            f"{base}/v1/jobs/export/export-if-range/download",
            {"Range": "bytes=0-31", "If-Range": etag},
        )
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert status_full == 200
    assert etag.startswith('"') and etag.endswith('"')
    assert headers_full["Accept-Ranges"] == "bytes"
    assert status_partial == 206
    assert headers_partial["ETag"] == etag
    assert headers_partial["Content-Range"] == f"bytes 0-31/{len(full)}"
    assert partial == full[:32]
    metrics = admin["jobs"]["exports"]["if_range_delivery"]["metrics"]
    assert metrics["if_range_checks"] == 1
    assert metrics["if_range_matches"] == 1
    assert metrics["if_range_mismatches"] == 0


def test_stage71_if_range_mismatch_returns_full_body(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_package(base)
        _, _, full = _download(f"{base}/v1/jobs/export/export-if-range/download")
        status, headers, body = _download(
            f"{base}/v1/jobs/export/export-if-range/download",
            {"Range": "bytes=0-31", "If-Range": '"stale"'},
        )
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert status == 200
    assert headers["X-Download-Mode"] == "if-range-mismatch-full"
    assert body == full
    metrics = admin["jobs"]["exports"]["if_range_delivery"]["metrics"]
    assert metrics["if_range_mismatches"] == 1


def test_stage71_invalid_range_returns_416_with_etag(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_package(base)
        package = _get(f"{base}/v1/jobs/export/export-if-range/package")["package"]
        try:
            _download(
                f"{base}/v1/jobs/export/export-if-range/download",
                {"Range": f"bytes={package['size_bytes'] + 1}-{package['size_bytes'] + 2}", "If-Range": f'"{package["sha256"]}"'},
            )
            raise AssertionError("invalid range should fail")
        except HTTPError as error:
            assert error.code == 416
            assert error.headers["ETag"] == f'"{package["sha256"]}"'
            assert error.headers["Content-Range"] == f"bytes */{package['size_bytes']}"
    finally:
        service.shutdown()


def test_stage71_smoke_passes(tmp_path) -> None:
    summary = run_stage71_if_range_package_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["etag_present"]
    assert summary["stage_gates"]["if_range_mismatch_full"]
