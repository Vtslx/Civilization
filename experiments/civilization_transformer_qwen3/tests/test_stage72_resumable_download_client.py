from __future__ import annotations

import json
from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import Stage66ExportConfig
from experiments.civilization_transformer_qwen3.analysis.stage67_export_lifecycle import Stage67ExportLifecycleConfig
from experiments.civilization_transformer_qwen3.analysis.stage68_export_package_delivery import Stage68PackageConfig
from experiments.civilization_transformer_qwen3.analysis.stage69_streaming_package_delivery import Stage69StreamingConfig
from experiments.civilization_transformer_qwen3.analysis.stage71_if_range_package_delivery import build_stage71_fake_service
from experiments.civilization_transformer_qwen3.analysis.stage72_resumable_download_client import (
    Stage72DownloadConfig,
    Stage72ResumableDownloadClient,
    run_stage72_resumable_download_smoke,
)
from experiments.civilization_transformer_qwen3.tests.test_stage71_if_range_package_delivery import _get, _payload, _post


def _service(tmp_path):
    return build_stage71_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
        streaming_config=Stage69StreamingConfig(download_chunk_bytes=32),
    )


def _create_package(base: str, export_id: str = "stage72-export") -> dict:
    submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("job-a"), _payload("job-b"), _payload("job-c")]})
    job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
    for job_id in job_ids:
        wait_for_job(base, job_id)
    _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": export_id})
    _post(f"{base}/v1/jobs/export/{export_id}/package", {})
    return _get(f"{base}/v1/jobs/export/{export_id}/package")["package"]


def test_stage72_full_download_verifies_sha256(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        package = _create_package(base)
        client = Stage72ResumableDownloadClient(Stage72DownloadConfig(chunk_bytes=32))
        target = tmp_path / "downloads" / "stage72.tar.gz"
        result = client.download(
            download_url=f"{base}/v1/jobs/export/stage72-export/download",
            package_url=f"{base}/v1/jobs/export/stage72-export/package",
            output_path=target,
        )
    finally:
        service.shutdown()

    assert result.status == "verified"
    assert result.expected_sha256 == package["sha256"]
    assert result.actual_sha256 == package["sha256"]
    assert target.exists()
    assert not Path(result.partial_path).exists()
    assert json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))["status"] == "verified"


def test_stage72_resume_uses_if_range_and_finishes(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        package = _create_package(base)
        client = Stage72ResumableDownloadClient(Stage72DownloadConfig(chunk_bytes=32))
        target = tmp_path / "downloads" / "resumed.tar.gz"
        partial = client.download(
            download_url=f"{base}/v1/jobs/export/stage72-export/download",
            package_url=f"{base}/v1/jobs/export/stage72-export/package",
            output_path=target,
            stop_after_bytes=64,
        )
        final = client.download(
            download_url=f"{base}/v1/jobs/export/stage72-export/download",
            package_url=f"{base}/v1/jobs/export/stage72-export/package",
            output_path=target,
        )
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert partial.status == "partial"
    assert final.status == "verified"
    assert final.actual_sha256 == package["sha256"]
    assert final.metrics["resume_attempts"] == 1
    assert final.metrics["resumed_downloads"] == 1
    assert admin["jobs"]["exports"]["if_range_delivery"]["metrics"]["if_range_matches"] >= 1


def test_stage72_stale_if_range_restarts_full_download(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        package = _create_package(base)
        client = Stage72ResumableDownloadClient(Stage72DownloadConfig(chunk_bytes=32))
        target = tmp_path / "downloads" / "stale.tar.gz"
        partial = client.download(
            download_url=f"{base}/v1/jobs/export/stage72-export/download",
            package_url=f"{base}/v1/jobs/export/stage72-export/package",
            output_path=target,
            stop_after_bytes=64,
        )
        sidecar = Path(partial.metadata_path)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["etag"] = '"stale"'
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        final = client.download(
            download_url=f"{base}/v1/jobs/export/stage72-export/download",
            package_url=f"{base}/v1/jobs/export/stage72-export/package",
            output_path=target,
        )
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert final.status == "verified"
    assert final.actual_sha256 == package["sha256"]
    assert final.metrics["stale_etag_restarts"] >= 1
    assert admin["jobs"]["exports"]["if_range_delivery"]["metrics"]["if_range_mismatches"] >= 1


def test_stage72_smoke_passes(tmp_path) -> None:
    summary = run_stage72_resumable_download_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["resume_verified"]
    assert summary["stage_gates"]["stale_restart_verified"]
