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
from .stage68_export_package_delivery import (
    DEFAULT_EXPORT_DIR as DEFAULT_STAGE68_EXPORT_DIR,
    DEFAULT_JOB_LOG as DEFAULT_STAGE68_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE68_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE68_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE68_RESULT_DIR,
    Stage68ExportPackageService,
    Stage68PackageConfig,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage69_streaming_package_delivery")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage69_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage69_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("artifacts/civilization/logs/stage69_job_results")
DEFAULT_EXPORT_DIR = Path("artifacts/civilization/logs/stage69_exports")


@dataclass(frozen=True)
class Stage69StreamingConfig:
    download_chunk_bytes: int = 64 * 1024


@dataclass
class Stage69StreamingMetrics:
    streaming_downloads: int = 0
    streamed_bytes: int = 0
    streamed_chunks: int = 0
    max_stream_chunk_bytes: int = 0
    last_stream_export_id: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage69StreamingPackageService(Stage68ExportPackageService):
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
        if streaming_config.download_chunk_bytes < 1:
            raise ValueError("download_chunk_bytes must be >= 1")
        self.streaming_config = streaming_config
        self.streaming_metrics = Stage69StreamingMetrics()
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
        )

    def get_streamable_export_package(self, export_id: str) -> tuple[dict[str, Any], Path] | None:
        package = self.get_export_package(export_id)
        if package is None or not package.get("exists") or not package.get("valid"):
            return None
        return package, Path(str(package["path"]))

    def record_stream_chunk(self, export_id: str, chunk_size: int) -> None:
        self.streaming_metrics.streamed_bytes += chunk_size
        self.streaming_metrics.streamed_chunks += 1
        self.streaming_metrics.max_stream_chunk_bytes = max(self.streaming_metrics.max_stream_chunk_bytes, chunk_size)
        self.streaming_metrics.last_stream_export_id = export_id

    def record_stream_download(self) -> None:
        self.streaming_metrics.streaming_downloads += 1
        self.package_metrics.package_downloads += 1

    def export_status(self) -> dict[str, Any]:
        payload = super().export_status()
        payload["streaming_delivery"] = {
            "config": asdict(self.streaming_config),
            "metrics": self.streaming_metrics.snapshot(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage69_streaming_package_delivery"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage69Handler(base_handler):
            server_version = "Stage69StreamingPackageQwenService/1.0"

            def _stream_package(self, package: dict[str, Any], path: Path) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(package["size_bytes"]))
                self.send_header("Content-Disposition", f"attachment; filename={package['export_id']}.tar.gz")
                self.send_header("X-Export-Id", str(package["export_id"]))
                self.send_header("X-Package-Sha256", str(package["sha256"]))
                self.send_header("X-Download-Mode", "streaming")
                self.end_headers()
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(service.streaming_config.download_chunk_bytes)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        service.record_stream_chunk(str(package["export_id"]), len(chunk))
                service.record_stream_download()

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
                    self._stream_package(metadata, path)
                    return
                super().do_GET()

        return Stage69Handler


def build_stage69_fake_service(
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
) -> Stage69StreamingPackageService:
    return Stage69StreamingPackageService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage69-fake"),
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


def build_stage69_real_service(
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
) -> Stage69StreamingPackageService:
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
    return Stage69StreamingPackageService(
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


def run_stage69_streaming_package_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage69_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(export_dir), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
        streaming_config=Stage69StreamingConfig(download_chunk_bytes=64),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage69-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "stage69-smoke-export"})
        packaged = _post_json(f"{base}/v1/jobs/export/stage69-smoke-export/package", {})
        package_info = _get_json(f"{base}/v1/jobs/export/stage69-smoke-export/package")
        admin_jobs_before = _get_json(f"{base}/admin/jobs")
        import urllib.request

        request = urllib.request.Request(f"{base}/v1/jobs/export/stage69-smoke-export/download")
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
            downloaded = response.read()
            headers = dict(response.headers)
        admin_jobs_after = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    package_path = export_dir / "stage69-smoke-export" / "package.tar.gz"
    streaming = admin_jobs_after.get("jobs", {}).get("exports", {}).get("streaming_delivery", {})
    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "package_created": packaged.get("status") == "created",
        "package_fetchable": package_info.get("package", {}).get("valid") is True,
        "download_streaming_header": headers.get("X-Download-Mode") == "streaming",
        "download_size_matches": len(downloaded) == package_path.stat().st_size,
        "streaming_metrics_visible": streaming.get("metrics", {}).get("streaming_downloads") == 1,
        "chunk_limit_respected": streaming.get("metrics", {}).get("max_stream_chunk_bytes", 0) <= 64,
    }
    summary = {
        "stage": "stage69_streaming_package_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "packaged": packaged,
        "package_info": package_info,
        "admin_jobs_before": admin_jobs_before,
        "admin_jobs_after": admin_jobs_after,
        "download_headers": headers,
        "downloaded_bytes": len(downloaded),
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
