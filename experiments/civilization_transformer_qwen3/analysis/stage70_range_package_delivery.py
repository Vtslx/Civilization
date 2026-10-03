from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .stage69_streaming_package_delivery import (
    DEFAULT_EXPORT_DIR as DEFAULT_STAGE69_EXPORT_DIR,
    DEFAULT_JOB_LOG as DEFAULT_STAGE69_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE69_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE69_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE69_RESULT_DIR,
    Stage69StreamingConfig,
    Stage69StreamingPackageService,
)
from experiments.civilization_transformer_qwen3.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage70_range_package_delivery")
DEFAULT_JOB_STATE = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage70_jobs_state.json")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage70_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage70_job_results")
DEFAULT_EXPORT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage70_exports")


@dataclass
class Stage70RangeMetrics:
    range_requests: int = 0
    partial_downloads: int = 0
    rejected_ranges: int = 0
    ranged_bytes: int = 0
    last_range_header: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage70RangePackageService(Stage69StreamingPackageService):
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
        self.range_metrics = Stage70RangeMetrics()
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

    def parse_range_header(self, value: str | None, size: int) -> tuple[int, int] | None:
        if value is None:
            return None
        self.range_metrics.range_requests += 1
        self.range_metrics.last_range_header = value
        if not value.startswith("bytes=") or "," in value:
            self.range_metrics.rejected_ranges += 1
            raise ValueError("unsupported range")
        spec = value.removeprefix("bytes=").strip()
        if "-" not in spec:
            self.range_metrics.rejected_ranges += 1
            raise ValueError("invalid range")
        start_text, end_text = spec.split("-", 1)
        try:
            if start_text == "":
                suffix = int(end_text)
                if suffix <= 0:
                    raise ValueError("invalid suffix range")
                start = max(0, size - suffix)
                end = size - 1
            else:
                start = int(start_text)
                end = size - 1 if end_text == "" else int(end_text)
        except ValueError:
            self.range_metrics.rejected_ranges += 1
            raise ValueError("invalid range") from None
        if start < 0 or end < start or start >= size:
            self.range_metrics.rejected_ranges += 1
            raise ValueError("unsatisfiable range")
        return start, min(end, size - 1)

    def record_partial_download(self, byte_count: int) -> None:
        self.range_metrics.partial_downloads += 1
        self.range_metrics.ranged_bytes += byte_count
        self.package_metrics.package_downloads += 1

    def export_status(self) -> dict[str, Any]:
        payload = super().export_status()
        payload["range_delivery"] = {
            "metrics": self.range_metrics.snapshot(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage70_range_package_delivery"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage70Handler(base_handler):
            server_version = "Stage70RangePackageQwenService/1.0"

            def _send_range_error(self, size: int) -> None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _stream_range(self, package: dict[str, Any], path: Path, start: int, end: int) -> None:
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Range", f"bytes {start}-{end}/{package['size_bytes']}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Disposition", f"attachment; filename={package['export_id']}.tar.gz")
                self.send_header("X-Export-Id", str(package["export_id"]))
                self.send_header("X-Package-Sha256", str(package["sha256"]))
                self.send_header("X-Download-Mode", "range-streaming")
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
                    range_header = self.headers.get("Range")
                    if range_header is None:
                        self._stream_package(metadata, path)
                        return
                    try:
                        parsed_range = service.parse_range_header(range_header, int(metadata["size_bytes"]))
                    except ValueError:
                        self._send_range_error(int(metadata["size_bytes"]))
                        return
                    if parsed_range is None:
                        self._stream_package(metadata, path)
                        return
                    start, end = parsed_range
                    self._stream_range(metadata, path, start, end)
                    return
                super().do_GET()

        return Stage70Handler


def build_stage70_fake_service(
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
) -> Stage70RangePackageService:
    return Stage70RangePackageService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage70-fake"),
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


def build_stage70_real_service(
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
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage70RangePackageService:
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
    return Stage70RangePackageService(
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


def run_stage70_range_package_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage70_fake_service(
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
        requests = [_payload(f"stage70-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "stage70-smoke-export"})
        packaged = _post_json(f"{base}/v1/jobs/export/stage70-smoke-export/package", {})
        package_info = _get_json(f"{base}/v1/jobs/export/stage70-smoke-export/package")
        import urllib.request

        request = urllib.request.Request(f"{base}/v1/jobs/export/stage70-smoke-export/download", headers={"Range": "bytes=0-31"})
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
            partial = response.read()
            headers = dict(response.headers)
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    range_metrics = admin_jobs.get("jobs", {}).get("exports", {}).get("range_delivery", {}).get("metrics", {})
    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "package_created": packaged.get("status") == "created",
        "package_fetchable": package_info.get("package", {}).get("valid") is True,
        "range_status_header": headers.get("Content-Range", "").startswith("bytes 0-31/"),
        "range_body_size": len(partial) == 32,
        "range_metrics_visible": range_metrics.get("partial_downloads") == 1,
    }
    summary = {
        "stage": "stage70_range_package_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "packaged": packaged,
        "package_info": package_info,
        "range_headers": headers,
        "partial_bytes": len(partial),
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
