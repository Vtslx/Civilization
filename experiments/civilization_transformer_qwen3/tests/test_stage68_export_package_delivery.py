from __future__ import annotations

from io import BytesIO
import json
import tarfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage61_async_jobs import Stage61JobConfig, wait_for_job
from experiments.civilization_transformer_qwen3.analysis.stage62_persistent_async_jobs import Stage62PersistenceConfig
from experiments.civilization_transformer_qwen3.analysis.stage63_job_retention_listing import Stage63RetentionConfig
from experiments.civilization_transformer_qwen3.analysis.stage64_external_result_store import Stage64ResultStoreConfig
from experiments.civilization_transformer_qwen3.analysis.stage65_batch_jobs import Stage65BatchConfig
from experiments.civilization_transformer_qwen3.analysis.stage66_batch_export import Stage66ExportConfig
from experiments.civilization_transformer_qwen3.analysis.stage67_export_lifecycle import Stage67ExportLifecycleConfig
from experiments.civilization_transformer_qwen3.analysis.stage68_export_package_delivery import (
    Stage68PackageConfig,
    build_stage68_fake_service,
    run_stage68_export_package_smoke,
)


def _payload(job_id: str) -> dict:
    return {
        "id": job_id,
        "text": "package secret text",
        "memory_items": ["package secret memory"],
        "rule_items": ["package secret rule"],
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


def _download(url: str) -> tuple[dict[str, str], bytes]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local test client
        return dict(response.headers), response.read()


def _service(tmp_path, *, max_package_bytes: int = 1024 * 1024):
    return build_stage68_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=max_package_bytes),
    )


def _create_export(base: str, export_id: str = "export-package") -> dict:
    submitted = _post(f"{base}/v1/jobs/batch", {"requests": [_payload("job-a"), _payload("job-b")]})
    job_ids = [item["job"]["job_id"] for item in submitted["batch"]["accepted"]]
    for job_id in job_ids:
        wait_for_job(base, job_id)
    return _post(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": export_id})


def test_stage68_creates_and_downloads_export_package(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_export(base, "export-package")
        packaged = _post(f"{base}/v1/jobs/export/export-package/package", {})
        package_info = _get(f"{base}/v1/jobs/export/export-package/package")
        headers, body = _download(f"{base}/v1/jobs/export/export-package/download")
        admin = _get(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    assert packaged["status"] == "created"
    assert package_info["package"]["valid"]
    assert headers["Content-Type"] == "application/gzip"
    assert headers["X-Package-Sha256"] == package_info["package"]["sha256"]
    with tarfile.open(fileobj=BytesIO(body), mode="r:gz") as archive:
        assert set(archive.getnames()) == {"manifest.json", "results.jsonl", "failures.jsonl"}
    assert admin["jobs"]["exports"]["package_delivery"]["metrics"]["package_downloads"] == 1


def test_stage68_rejects_packaging_tampered_export(tmp_path) -> None:
    service = _service(tmp_path)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_export(base, "export-tamper")
        results = tmp_path / "exports" / "export-tamper" / "results.jsonl"
        results.write_text(results.read_text(encoding="utf-8") + "\n{\"tampered\": true}\n", encoding="utf-8")
        try:
            _post(f"{base}/v1/jobs/export/export-tamper/package", {})
            raise AssertionError("tampered export should not package")
        except HTTPError as error:
            assert error.code == 409
    finally:
        service.shutdown()


def test_stage68_rejects_oversized_package(tmp_path) -> None:
    service = _service(tmp_path, max_package_bytes=1)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _create_export(base, "export-large")
        try:
            _post(f"{base}/v1/jobs/export/export-large/package", {})
            raise AssertionError("oversized package should fail")
        except HTTPError as error:
            assert error.code == 413
    finally:
        service.shutdown()


def test_stage68_smoke_passes(tmp_path) -> None:
    summary = run_stage68_export_package_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["package_created"]
    assert summary["stage_gates"]["package_file_written"]
