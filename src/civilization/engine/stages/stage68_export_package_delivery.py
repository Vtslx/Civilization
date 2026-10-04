from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tarfile
import time
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
from .stage67_export_lifecycle import (
    DEFAULT_EXPORT_DIR as DEFAULT_STAGE67_EXPORT_DIR,
    DEFAULT_JOB_LOG as DEFAULT_STAGE67_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE67_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE67_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE67_RESULT_DIR,
    Stage67ExportLifecycleConfig,
    Stage67ExportLifecycleService,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage68_export_package_delivery")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage68_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage68_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("artifacts/civilization/logs/stage68_job_results")
DEFAULT_EXPORT_DIR = Path("artifacts/civilization/logs/stage68_exports")


@dataclass(frozen=True)
class Stage68PackageConfig:
    package_suffix: str = ".tar.gz"
    max_package_bytes: int = 512 * 1024 * 1024


@dataclass
class Stage68PackageMetrics:
    packages_created: int = 0
    package_downloads: int = 0
    package_integrity_checks: int = 0
    last_package_export_id: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage68ExportPackageService(Stage67ExportLifecycleService):
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
    ) -> None:
        if package_config.max_package_bytes < 1:
            raise ValueError("max_package_bytes must be >= 1")
        if not package_config.package_suffix:
            raise ValueError("package_suffix must be non-empty")
        self.package_config = package_config
        self.package_metrics = Stage68PackageMetrics()
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
        )

    def _package_path(self, export_id: str) -> Path:
        return self._export_path(export_id) / f"package{self.package_config.package_suffix}"

    def _package_metadata(self, export_id: str, package_path: Path) -> dict[str, Any]:
        return {
            "export_id": export_id,
            "path": str(package_path),
            "exists": package_path.exists(),
            "size_bytes": package_path.stat().st_size if package_path.exists() else 0,
            "sha256": self._file_sha256(package_path) if package_path.exists() else None,
            "created_at": time.time(),
        }

    def create_export_package(self, export_id: str) -> dict[str, Any] | None:
        manifest = self.get_export_manifest(export_id)
        if manifest is None:
            return None
        integrity = self.check_export_integrity(export_id)
        if not integrity or not integrity.get("valid"):
            raise ValueError("export integrity check failed")
        files = manifest.get("files", {})
        manifest_path = Path(files.get("manifest", self._export_path(export_id) / "manifest.json"))
        results_path = Path(files.get("results", ""))
        failures_path = Path(files.get("failures", ""))
        package_path = self._package_path(export_id)
        temp_path = package_path.with_suffix(package_path.suffix + ".tmp")
        package_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(temp_path, "w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            archive.add(results_path, arcname="results.jsonl")
            archive.add(failures_path, arcname="failures.jsonl")
        if temp_path.stat().st_size > self.package_config.max_package_bytes:
            temp_path.unlink(missing_ok=True)
            raise OverflowError("export package exceeds max_package_bytes")
        temp_path.replace(package_path)
        package = self._package_metadata(export_id, package_path)
        manifest["package"] = package
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.package_metrics.packages_created += 1
        self.package_metrics.last_package_export_id = export_id
        self._append_job_event(
            {
                "event": "export_package_created",
                "export_id": export_id,
                "size_bytes": package["size_bytes"],
                "sha256": package["sha256"],
            }
        )
        return package

    def get_export_package(self, export_id: str) -> dict[str, Any] | None:
        manifest = self.get_export_manifest(export_id)
        if manifest is None:
            return None
        package = manifest.get("package")
        if isinstance(package, dict) and package.get("path"):
            path = Path(str(package["path"]))
        else:
            path = self._package_path(export_id)
        if not path.exists():
            return {
                "export_id": export_id,
                "exists": False,
                "path": str(path),
                "size_bytes": 0,
                "sha256": None,
                "valid": False,
            }
        actual = self._package_metadata(export_id, path)
        expected_sha = package.get("sha256") if isinstance(package, dict) else None
        expected_size = package.get("size_bytes") if isinstance(package, dict) else None
        self.package_metrics.package_integrity_checks += 1
        return {
            **actual,
            "valid": expected_sha == actual["sha256"] and expected_size == actual["size_bytes"],
            "expected_sha256": expected_sha,
            "expected_size_bytes": expected_size,
        }

    def read_export_package_bytes(self, export_id: str) -> tuple[dict[str, Any], bytes] | None:
        package = self.get_export_package(export_id)
        if package is None or not package.get("exists") or not package.get("valid"):
            return None
        path = Path(str(package["path"]))
        data = path.read_bytes()
        self.package_metrics.package_downloads += 1
        return package, data

    def export_status(self) -> dict[str, Any]:
        payload = super().export_status()
        payload["package_delivery"] = {
            "config": asdict(self.package_config),
            "metrics": self.package_metrics.snapshot(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage68_export_package_delivery"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage68Handler(base_handler):
            server_version = "Stage68ExportPackageQwenService/1.0"

            def _send_package(self, package: dict[str, Any], body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", f"attachment; filename={package['export_id']}.tar.gz")
                self.send_header("X-Export-Id", str(package["export_id"]))
                self.send_header("X-Package-Sha256", str(package["sha256"]))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path.startswith("/v1/jobs/export/") and parsed.path.endswith("/package"):
                    if not self._check_access():
                        return
                    export_id = parsed.path.split("/")[-2]
                    package = service.get_export_package(export_id)
                    if package is None:
                        self._send_json(404, {"status": "error", "error": "export not found"})
                        return
                    self._send_json(200, {"status": "ok", "package": package})
                    return
                if parsed.path.startswith("/v1/jobs/export/") and parsed.path.endswith("/download"):
                    if not self._check_access():
                        return
                    export_id = parsed.path.split("/")[-2]
                    package = service.read_export_package_bytes(export_id)
                    if package is None:
                        self._send_json(404, {"status": "error", "error": "package not found or invalid"})
                        return
                    metadata, body = package
                    self._send_package(metadata, body)
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path.startswith("/v1/jobs/export/") and parsed.path.endswith("/package"):
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(400, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                    if not self._check_access(payload):
                        return
                    export_id = parsed.path.split("/")[-2]
                    try:
                        package = service.create_export_package(export_id)
                    except OverflowError as error:
                        self._send_json(413, {"status": "error", "error_type": "PackageTooLarge", "error": str(error)})
                        return
                    except ValueError as error:
                        self._send_json(409, {"status": "error", "error_type": "InvalidExportIntegrity", "error": str(error)})
                        return
                    if package is None:
                        self._send_json(404, {"status": "error", "error": "export not found"})
                        return
                    self._send_json(201, {"status": "created", "package": package})
                    return
                super().do_POST()

        return Stage68Handler


def build_stage68_fake_service(
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
    runtime_delay_seconds: float = 0.0,
) -> Stage68ExportPackageService:
    return Stage68ExportPackageService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage68-fake"),
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
    )


def build_stage68_real_service(
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
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage68ExportPackageService:
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
    return Stage68ExportPackageService(
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
    )


def run_stage68_export_package_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage68_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(export_dir), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage68-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "stage68-smoke-export"})
        packaged = _post_json(f"{base}/v1/jobs/export/stage68-smoke-export/package", {})
        package_info = _get_json(f"{base}/v1/jobs/export/stage68-smoke-export/package")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    package_path = export_dir / "stage68-smoke-export" / "package.tar.gz"
    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "package_created": packaged.get("status") == "created",
        "package_file_written": package_path.exists() and package_path.stat().st_size > 0,
        "package_fetchable": package_info.get("package", {}).get("valid") is True,
        "package_metrics_visible": "package_delivery" in admin_jobs.get("jobs", {}).get("exports", {}),
    }
    summary = {
        "stage": "stage68_export_package_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "packaged": packaged,
        "package_info": package_info,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
