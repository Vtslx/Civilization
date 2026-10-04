from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import Stage61JobConfig, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import Stage65BatchConfig
from .stage66_batch_export import (
    DEFAULT_EXPORT_DIR as DEFAULT_STAGE66_EXPORT_DIR,
    DEFAULT_JOB_LOG as DEFAULT_STAGE66_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE66_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE66_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE66_RESULT_DIR,
    Stage66BatchExportService,
    Stage66ExportConfig,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage67_export_lifecycle")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage67_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage67_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("artifacts/civilization/logs/stage67_job_results")
DEFAULT_EXPORT_DIR = Path("artifacts/civilization/logs/stage67_exports")


@dataclass(frozen=True)
class Stage67ExportLifecycleConfig:
    export_ttl_seconds: float = 7 * 24 * 3600
    max_persisted_exports: int = 1000
    max_export_list_limit: int = 200


@dataclass
class Stage67ExportLifecycleMetrics:
    integrity_checks: int = 0
    cleanup_runs: int = 0
    exports_deleted: int = 0
    last_cleanup_deleted: int = 0

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage67ExportLifecycleService(Stage66BatchExportService):
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
    ) -> None:
        if lifecycle_config.max_persisted_exports < 1:
            raise ValueError("max_persisted_exports must be >= 1")
        if lifecycle_config.max_export_list_limit < 1:
            raise ValueError("max_export_list_limit must be >= 1")
        if lifecycle_config.export_ttl_seconds < 0:
            raise ValueError("export_ttl_seconds must be >= 0")
        self.lifecycle_config = lifecycle_config
        self.lifecycle_metrics = Stage67ExportLifecycleMetrics()
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
        )

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _file_integrity(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": self._file_sha256(path) if path.exists() else None,
        }

    def _load_manifest_path(self, export_path: Path) -> tuple[Path, dict[str, Any] | None]:
        manifest_path = export_path / "manifest.json"
        if not manifest_path.exists():
            return manifest_path, None
        try:
            return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return manifest_path, None

    def _export_summary_from_path(self, export_path: Path) -> dict[str, Any] | None:
        manifest_path, manifest = self._load_manifest_path(export_path)
        if manifest is None:
            return None
        created_at = float(manifest.get("created_at", manifest_path.stat().st_mtime))
        return {
            "export_id": str(manifest.get("export_id", export_path.name)),
            "created_at": created_at,
            "job_count": int(manifest.get("job_count", 0)),
            "exported_jobs": int(manifest.get("exported_jobs", 0)),
            "exported_rows": int(manifest.get("exported_rows", 0)),
            "failures": int(manifest.get("failures", 0)),
            "path": str(export_path),
        }

    def _write_integrity_to_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        files = manifest.get("files", {})
        results = Path(files.get("results", ""))
        failures = Path(files.get("failures", ""))
        integrity = {
            "results": self._file_integrity(results),
            "failures": self._file_integrity(failures),
            "created_at": time.time(),
        }
        manifest["integrity"] = integrity
        manifest_path = Path(files.get("manifest", self._export_path(str(manifest["export_id"])) / "manifest.json"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def create_export(self, job_ids: list[str], *, export_id: str | None = None) -> dict[str, Any]:
        manifest = super().create_export(job_ids, export_id=export_id)
        return self._write_integrity_to_manifest(manifest)

    def list_exports(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        root = self._export_root()
        root.mkdir(parents=True, exist_ok=True)
        summaries = [item for item in (self._export_summary_from_path(path) for path in root.iterdir() if path.is_dir()) if item]
        summaries.sort(key=lambda item: (float(item["created_at"]), str(item["export_id"])), reverse=True)
        offset = max(0, offset)
        limit = max(1, min(limit, self.lifecycle_config.max_export_list_limit))
        return {
            "offset": offset,
            "limit": limit,
            "total": len(summaries),
            "exports": summaries[offset : offset + limit],
        }

    def check_export_integrity(self, export_id: str) -> dict[str, Any] | None:
        manifest = self.get_export_manifest(export_id)
        if manifest is None:
            return None
        files = manifest.get("files", {})
        manifest_path = Path(files.get("manifest", self._export_path(export_id) / "manifest.json"))
        results = Path(files.get("results", ""))
        failures = Path(files.get("failures", ""))
        expected = manifest.get("integrity", {})
        actual = {
            "manifest": self._file_integrity(manifest_path),
            "results": self._file_integrity(results),
            "failures": self._file_integrity(failures),
        }
        checks = {
            "results_sha256": expected.get("results", {}).get("sha256") == actual["results"]["sha256"],
            "failures_sha256": expected.get("failures", {}).get("sha256") == actual["failures"]["sha256"],
            "results_size": expected.get("results", {}).get("size_bytes") == actual["results"]["size_bytes"],
            "failures_size": expected.get("failures", {}).get("size_bytes") == actual["failures"]["size_bytes"],
        }
        self.lifecycle_metrics.integrity_checks += 1
        return {
            "export_id": export_id,
            "valid": all(checks.values()),
            "checks": checks,
            "expected": expected,
            "actual": actual,
        }

    def cleanup_exports(self, *, ttl_seconds: float | None = None, max_exports: int | None = None) -> dict[str, Any]:
        root = self._export_root()
        root.mkdir(parents=True, exist_ok=True)
        ttl = self.lifecycle_config.export_ttl_seconds if ttl_seconds is None else max(0.0, ttl_seconds)
        keep = self.lifecycle_config.max_persisted_exports if max_exports is None else max(1, max_exports)
        now = time.time()
        summaries = [item for item in (self._export_summary_from_path(path) for path in root.iterdir() if path.is_dir()) if item]
        summaries.sort(key=lambda item: (float(item["created_at"]), str(item["export_id"])), reverse=True)

        delete_ids: set[str] = set()
        for item in summaries:
            if ttl == 0 or now - float(item["created_at"]) > ttl:
                delete_ids.add(str(item["export_id"]))
        for item in summaries[keep:]:
            delete_ids.add(str(item["export_id"]))

        deleted: list[str] = []
        for export_id in sorted(delete_ids):
            path = self._export_path(export_id)
            if path.exists():
                shutil.rmtree(path)
                deleted.append(export_id)

        self.lifecycle_metrics.cleanup_runs += 1
        self.lifecycle_metrics.exports_deleted += len(deleted)
        self.lifecycle_metrics.last_cleanup_deleted = len(deleted)
        self._append_job_event({"event": "exports_cleanup", "deleted": len(deleted), "export_ids": deleted})
        return {
            "deleted": len(deleted),
            "export_ids": deleted,
            "ttl_seconds": ttl,
            "max_exports": keep,
        }

    def export_status(self) -> dict[str, Any]:
        payload = super().export_status()
        payload["lifecycle"] = {
            "config": asdict(self.lifecycle_config),
            "metrics": self.lifecycle_metrics.snapshot(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage67_export_lifecycle"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage67Handler(base_handler):
            server_version = "Stage67ExportLifecycleQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/v1/jobs/exports":
                    if not self._check_access():
                        return
                    query = parse_qs(parsed.query)
                    try:
                        offset = int(query.get("offset", ["0"])[0])
                        limit = int(query.get("limit", ["50"])[0])
                    except (TypeError, ValueError):
                        self._send_json(400, {"status": "error", "error": "offset and limit must be integers"})
                        return
                    self._send_json(200, {"status": "ok", "exports": service.list_exports(offset=offset, limit=limit)})
                    return
                if parsed.path.startswith("/v1/jobs/export/") and parsed.path.endswith("/integrity"):
                    if not self._check_access():
                        return
                    export_id = parsed.path.split("/")[-2]
                    integrity = service.check_export_integrity(export_id)
                    if integrity is None:
                        self._send_json(404, {"status": "error", "error": "export not found"})
                        return
                    self._send_json(200, {"status": "ok", "integrity": integrity})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/v1/jobs/exports/cleanup":
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(400, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                    if not self._check_access(payload):
                        return
                    try:
                        ttl = payload.get("ttl_seconds") if isinstance(payload, dict) else None
                        max_exports = payload.get("max_exports") if isinstance(payload, dict) else None
                        cleanup = service.cleanup_exports(
                            ttl_seconds=float(ttl) if ttl is not None else None,
                            max_exports=int(max_exports) if max_exports is not None else None,
                        )
                    except (TypeError, ValueError) as error:
                        self._send_json(400, {"status": "error", "error": str(error)})
                        return
                    self._send_json(200, {"status": "ok", "cleanup": cleanup})
                    return
                super().do_POST()

        return Stage67Handler


def build_stage67_fake_service(
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
    runtime_delay_seconds: float = 0.0,
) -> Stage67ExportLifecycleService:
    return Stage67ExportLifecycleService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage67-fake"),
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
    )


def build_stage67_real_service(
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
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage67ExportLifecycleService:
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
    return Stage67ExportLifecycleService(
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
    )


def run_stage67_export_lifecycle_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage67_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(export_dir), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=2, max_export_list_limit=10),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage67-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported_a = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids[:2], "export_id": "stage67-smoke-export-a"})
        exported_b = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids[1:], "export_id": "stage67-smoke-export-b"})
        listed = _get_json(f"{base}/v1/jobs/exports?offset=0&limit=10")
        integrity = _get_json(f"{base}/v1/jobs/export/stage67-smoke-export-a/integrity")
        cleanup = _post_json(f"{base}/v1/jobs/exports/cleanup", {"ttl_seconds": 3600, "max_exports": 1})
        listed_after = _get_json(f"{base}/v1/jobs/exports?offset=0&limit=10")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "exports_created": exported_a.get("status") == "created" and exported_b.get("status") == "created",
        "exports_listed": listed.get("exports", {}).get("total") == 2,
        "integrity_valid": integrity.get("integrity", {}).get("valid") is True,
        "cleanup_deleted": cleanup.get("cleanup", {}).get("deleted") == 1,
        "cleanup_visible": listed_after.get("exports", {}).get("total") == 1,
        "lifecycle_metrics_visible": "lifecycle" in admin_jobs.get("jobs", {}).get("exports", {}),
    }
    summary = {
        "stage": "stage67_export_lifecycle_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": [exported_a, exported_b],
        "listed": listed,
        "integrity": integrity,
        "cleanup": cleanup,
        "listed_after": listed_after,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
