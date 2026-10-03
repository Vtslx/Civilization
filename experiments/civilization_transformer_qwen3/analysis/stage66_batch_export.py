from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
import uuid

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import Stage61JobConfig, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import (
    DEFAULT_JOB_LOG as DEFAULT_STAGE65_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE65_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE65_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE65_RESULT_DIR,
    Stage65BatchConfig,
    Stage65BatchJobService,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage66_batch_export")
DEFAULT_JOB_STATE = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage66_jobs_state.json")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage66_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage66_job_results")
DEFAULT_EXPORT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage66_exports")


@dataclass(frozen=True)
class Stage66ExportConfig:
    export_dir: str = str(DEFAULT_EXPORT_DIR)
    max_export_jobs: int = 256


@dataclass
class Stage66ExportMetrics:
    exports_created: int = 0
    exported_jobs: int = 0
    exported_rows: int = 0
    export_failures: int = 0
    last_export_id: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage66BatchExportService(Stage65BatchJobService):
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
        export_config: Stage66ExportConfig = Stage66ExportConfig(),
    ) -> None:
        if export_config.max_export_jobs < 1:
            raise ValueError("max_export_jobs must be >= 1")
        self.export_config = export_config
        self.export_metrics = Stage66ExportMetrics()
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
        )

    def _export_root(self) -> Path:
        return Path(self.export_config.export_dir)

    def _export_path(self, export_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in export_id)
        return self._export_root() / safe

    def create_export(self, job_ids: list[str], *, export_id: str | None = None) -> dict[str, Any]:
        if len(job_ids) > self.export_config.max_export_jobs:
            raise OverflowError(f"export contains {len(job_ids)} job ids; max_export_jobs={self.export_config.max_export_jobs}")
        export_id = export_id or f"export-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        output = self._export_path(export_id)
        output.mkdir(parents=True, exist_ok=True)
        results_path = output / "results.jsonl"
        failures_path = output / "failures.jsonl"
        manifest_path = output / "manifest.json"
        exported_rows = 0
        failures = 0
        exported_jobs = 0
        job_entries: list[dict[str, Any]] = []

        with results_path.open("w", encoding="utf-8") as results_handle, failures_path.open("w", encoding="utf-8") as failures_handle:
            for job_id in [str(item) for item in job_ids]:
                record = self.get_job(job_id)
                if record is None:
                    failures += 1
                    failure = {"job_id": job_id, "error": "job_not_found"}
                    failures_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    job_entries.append({"job_id": job_id, "status": "not_found", "rows": 0})
                    continue
                page = self.read_result_page(job_id, offset=0, limit=self.result_store_config.max_result_page_limit)
                if not page or not page.get("result_available"):
                    failures += 1
                    failure = {"job_id": job_id, "status": record.status, "error": "result_unavailable"}
                    failures_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    job_entries.append({"job_id": job_id, "status": record.status, "rows": 0})
                    continue
                total_rows = int(page.get("total_rows", 0))
                offset = 0
                while offset < total_rows:
                    current = self.read_result_page(job_id, offset=offset, limit=self.result_store_config.max_result_page_limit)
                    rows = current.get("rows", []) if current else []
                    if not rows:
                        break
                    for row in rows:
                        results_handle.write(json.dumps({"job_id": job_id, "row": row}, ensure_ascii=False) + "\n")
                        exported_rows += 1
                    offset += len(rows)
                exported_jobs += 1
                job_entries.append({"job_id": job_id, "status": record.status, "rows": total_rows})

        manifest = {
            "stage": "stage66_batch_export",
            "export_id": export_id,
            "created_at": time.time(),
            "job_ids": [str(item) for item in job_ids],
            "job_count": len(job_ids),
            "exported_jobs": exported_jobs,
            "exported_rows": exported_rows,
            "failures": failures,
            "files": {
                "manifest": str(manifest_path),
                "results": str(results_path),
                "failures": str(failures_path),
            },
            "jobs": job_entries,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.export_metrics.exports_created += 1
        self.export_metrics.exported_jobs += exported_jobs
        self.export_metrics.exported_rows += exported_rows
        self.export_metrics.export_failures += failures
        self.export_metrics.last_export_id = export_id
        self._append_job_event(
            {
                "event": "jobs_export_created",
                "export_id": export_id,
                "job_count": len(job_ids),
                "exported_jobs": exported_jobs,
                "exported_rows": exported_rows,
                "failures": failures,
            }
        )
        return manifest

    def get_export_manifest(self, export_id: str) -> dict[str, Any] | None:
        manifest = self._export_path(export_id) / "manifest.json"
        if not manifest.exists():
            return None
        return json.loads(manifest.read_text(encoding="utf-8"))

    def export_status(self) -> dict[str, Any]:
        return {
            "config": asdict(self.export_config),
            "metrics": self.export_metrics.snapshot(),
        }

    def job_status(self) -> dict[str, Any]:
        payload = super().job_status()
        payload["exports"] = self.export_status()
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage66_batch_export"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage66Handler(base_handler):
            server_version = "Stage66BatchExportQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path.startswith("/v1/jobs/export/"):
                    if not self._check_access():
                        return
                    export_id = parsed.path.rsplit("/", 1)[-1]
                    manifest = service.get_export_manifest(export_id)
                    if manifest is None:
                        self._send_json(404, {"status": "error", "error": "export not found"})
                        return
                    self._send_json(200, {"status": "ok", "export": manifest})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/v1/jobs/export":
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(400, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                    if not self._check_access(payload):
                        return
                    job_ids = payload.get("job_ids") if isinstance(payload, dict) else None
                    if not isinstance(job_ids, list):
                        self._send_json(400, {"status": "error", "error": "job_ids must be a list"})
                        return
                    try:
                        manifest = service.create_export(job_ids, export_id=payload.get("export_id"))
                    except OverflowError as error:
                        self._send_json(429, {"status": "error", "error_type": "ExportTooLarge", "error": str(error)})
                        return
                    self._send_json(201, {"status": "created", "export": manifest})
                    return
                super().do_POST()

        return Stage66Handler


def build_stage66_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage66BatchExportService:
    return Stage66BatchExportService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage66-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
    )


def build_stage66_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B",
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage66BatchExportService:
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
    return Stage66BatchExportService(
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
    )


def run_stage66_batch_export_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    export_dir = output / "exports"
    service = build_stage66_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(export_dir), max_export_jobs=4),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage66-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": "stage66-smoke-export"})
        fetched = _get_json(f"{base}/v1/jobs/export/stage66-smoke-export")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    manifest_path = export_dir / "stage66-smoke-export" / "manifest.json"
    results_path = export_dir / "stage66-smoke-export" / "results.jsonl"
    failures_path = export_dir / "stage66-smoke-export" / "failures.jsonl"
    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "manifest_written": manifest_path.exists(),
        "results_written": results_path.exists() and results_path.stat().st_size > 0,
        "failures_written": failures_path.exists(),
        "export_fetchable": fetched.get("export", {}).get("export_id") == "stage66-smoke-export",
        "export_metrics_visible": "exports" in admin_jobs.get("jobs", {}),
    }
    summary = {
        "stage": "stage66_batch_export_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "fetched": fetched,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
