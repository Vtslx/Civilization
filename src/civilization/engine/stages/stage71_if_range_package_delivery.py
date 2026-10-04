from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import Stage61JobConfig, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import Stage65BatchConfig
from .stage66_batch_export import Stage66ExportConfig
from .stage67_export_lifecycle import Stage67ExportLifecycleConfig
from .stage68_export_package_delivery import Stage68PackageConfig
from .stage69_streaming_package_delivery import Stage69StreamingConfig
from .stage70_range_package_delivery import (
    DEFAULT_EXPORT_DIR as DEFAULT_STAGE70_EXPORT_DIR,
    DEFAULT_JOB_LOG as DEFAULT_STAGE70_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE70_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE70_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE70_RESULT_DIR,
    Stage70RangePackageService,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage71_if_range_package_delivery")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage71_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage71_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("artifacts/civilization/logs/stage71_job_results")
DEFAULT_EXPORT_DIR = Path("artifacts/civilization/logs/stage71_exports")


@dataclass
class Stage71IfRangeMetrics:
    etag_downloads: int = 0
    if_range_checks: int = 0
    if_range_matches: int = 0
    if_range_mismatches: int = 0
    last_etag: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage71IfRangePackageService(Stage70RangePackageService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
        queue_config: Stage60QueueConfig = Stage60QueueConfig(),
        job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
        persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
        retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
        result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
        batch_config: Stage65BatchConfig = Stage65BatchConfig(),
        export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
        lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
        package_config: Stage68PackageConfig = Stage68PackageConfig(),
        streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
    ) -> None:
        self.if_range_metrics = Stage71IfRangeMetrics()
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
            job_config=job_config,
            persistence_config=persistence_config,
            retention_config=retention_config,
            result_store_config=result_store_config,
            batch_config=batch_config,
            export_config=export_config,
            lifecycle_config=lifecycle_config,
            package_config=package_config,
            streaming_config=streaming_config,
        )

    def strong_etag(self, package: dict[str, Any]) -> str:
        return f"\"{package['sha256']}\""

    def if_range_matches(self, header: str | None, etag: str) -> bool:
        if header is None:
            return True
        self.if_range_metrics.if_range_checks += 1
        normalized = header.strip()
        matched = normalized == etag or normalized == etag.strip('"')
        if matched:
            self.if_range_metrics.if_range_matches += 1
        else:
            self.if_range_metrics.if_range_mismatches += 1
        return matched

    def record_etag_download(self, etag: str) -> None:
        self.if_range_metrics.etag_downloads += 1
        self.if_range_metrics.last_etag = etag

    def export_status(self) -> dict[str, Any]:
        payload = super().export_status()
        payload["if_range_delivery"] = {
            "metrics": self.if_range_metrics.snapshot(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage71_if_range_package_delivery"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage71Handler(base_handler):
            server_version = "Stage71IfRangePackageQwenService/1.0"

            def _stream_full_with_etag(self, package: dict[str, Any], path: Path, *, mode: str = "streaming") -> None:
                etag = service.strong_etag(package)
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(package["size_bytes"]))
                self.send_header("Content-Disposition", f"attachment; filename={package['export_id']}.tar.gz")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", etag)
                self.send_header("X-Export-Id", str(package["export_id"]))
                self.send_header("X-Package-Sha256", str(package["sha256"]))
                self.send_header("X-Download-Mode", mode)
                self.end_headers()
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(service.streaming_config.download_chunk_bytes)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        service.record_stream_chunk(str(package["export_id"]), len(chunk))
                service.record_stream_download()
                service.record_etag_download(etag)

            def _stream_range_with_etag(self, package: dict[str, Any], path: Path, start: int, end: int) -> None:
                etag = service.strong_etag(package)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Range", f"bytes {start}-{end}/{package['size_bytes']}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", etag)
                self.send_header("Content-Disposition", f"attachment; filename={package['export_id']}.tar.gz")
                self.send_header("X-Export-Id", str(package["export_id"]))
                self.send_header("X-Package-Sha256", str(package["sha256"]))
                self.send_header("X-Download-Mode", "if-range-streaming")
                self.end_headers()
                remaining = length
                with path.open("rb") as handle:
                    handle.seek(start)
                    while remaining > 0:
                        chunk = handle.read(min(service.streaming_config.download_chunk_bytes, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        service.record_stream_chunk(str(package["export_id"]), len(chunk))
                        remaining -= len(chunk)
                service.record_partial_download(length)
                service.record_etag_download(etag)

            def _send_range_error_with_etag(self, package: dict[str, Any]) -> None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{package['size_bytes']}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", service.strong_etag(package))
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path.startswith("/v1/jobs/export/") and parsed.path.endswith("/download"):
                    if not self._check_access():
                        return
                    export_id = parsed.path.split("/")[-2]
                    package = service.get_streamable_export_package(export_id)
                    if package is None:
                        self._send_json(404, {"status": "error", "error": "package not found or invalid"})
                        return
                    metadata, path = package
                    etag = service.strong_etag(metadata)
                    range_header = self.headers.get("Range")
                    if range_header is None:
                        self._stream_full_with_etag(metadata, path)
                        return
                    if not service.if_range_matches(self.headers.get("If-Range"), etag):
                        self._stream_full_with_etag(metadata, path, mode="if-range-mismatch-full")
                        return
                    try:
                        parsed_range = service.parse_range_header(range_header, int(metadata["size_bytes"]))
                    except ValueError:
                        self._send_range_error_with_etag(metadata)
                        return
                    if parsed_range is None:
                        self._stream_full_with_etag(metadata, path)
                        return
                    start, end = parsed_range
                    self._stream_range_with_etag(metadata, path, start, end)
                    return
                super().do_GET()

        return Stage71Handler


def build_stage71_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
    lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
    package_config: Stage68PackageConfig = Stage68PackageConfig(),
    streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage71IfRangePackageService:
    return Stage71IfRangePackageService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage71-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
        lifecycle_config=lifecycle_config,
        package_config=package_config,
        streaming_config=streaming_config,
    )


def build_stage71_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
    lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
    package_config: Stage68PackageConfig = Stage68PackageConfig(),
    streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage71IfRangePackageService:
    base = build_stage60_real_service(
        port=port,
        security=security,
        queue_config=queue_config,
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
    )
    return Stage71IfRangePackageService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
        lifecycle_config=lifecycle_config,
        package_config=package_config,
        streaming_config=streaming_config,
    )


def run_stage71_if_range_package_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage71_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(export_dir), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
        streaming_config=Stage69StreamingConfig(download_chunk_bytes=32),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage71-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "stage71-smoke-export"})
        packaged = _post_json(f"{base}/v1/jobs/export/stage71-smoke-export/package", {})
        import urllib.request

        full_request = urllib.request.Request(f"{base}/v1/jobs/export/stage71-smoke-export/download")
        with urllib.request.urlopen(full_request, timeout=10) as response:  # noqa: S310 - local smoke client
            full = response.read()
            full_headers = dict(response.headers)
        etag = full_headers["ETag"]
        range_request = urllib.request.Request(
            f"{base}/v1/jobs/export/stage71-smoke-export/download",
            headers={"Range": "bytes=0-31", "If-Range": etag},
        )
        with urllib.request.urlopen(range_request, timeout=10) as response:  # noqa: S310 - local smoke client
            partial = response.read()
            range_headers = dict(response.headers)
        mismatch_request = urllib.request.Request(
            f"{base}/v1/jobs/export/stage71-smoke-export/download",
            headers={"Range": "bytes=0-31", "If-Range": "\"not-current\""},
        )
        with urllib.request.urlopen(mismatch_request, timeout=10) as response:  # noqa: S310 - local smoke client
            mismatch = response.read()
            mismatch_headers = dict(response.headers)
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    metrics = admin_jobs.get("jobs", {}).get("exports", {}).get("if_range_delivery", {}).get("metrics", {})
    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "package_created": packaged.get("status") == "created",
        "etag_present": full_headers.get("ETag", "").startswith('"'),
        "if_range_match_partial": range_headers.get("Content-Range", "").startswith("bytes 0-31/") and partial == full[:32],
        "if_range_mismatch_full": mismatch_headers.get("X-Download-Mode") == "if-range-mismatch-full" and mismatch == full,
        "if_range_metrics_visible": metrics.get("if_range_matches") == 1 and metrics.get("if_range_mismatches") == 1,
    }
    summary = {
        "stage": "stage71_if_range_package_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "packaged": packaged,
        "full_headers": full_headers,
        "range_headers": range_headers,
        "mismatch_headers": mismatch_headers,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
