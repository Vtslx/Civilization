from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import Stage61JobConfig, Stage61JobRecord, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import (
    DEFAULT_JOB_LOG as DEFAULT_STAGE63_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE63_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE63_OUTPUT_DIR,
    Stage63JobRetentionListingService,
    Stage63RetentionConfig,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage64_external_result_store")
DEFAULT_JOB_STATE = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage64_jobs_state.json")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage64_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage64_job_results")


@dataclass(frozen=True)
class Stage64ResultStoreConfig:
    result_dir: str = str(DEFAULT_RESULT_DIR)
    inline_result_row_limit: int = 0
    max_result_page_limit: int = 200
    delete_result_file_on_job_prune: bool = True


@dataclass
class Stage64ResultStoreMetrics:
    result_files_written: int = 0
    result_files_deleted: int = 0
    result_rows_written: int = 0
    result_bytes_written: int = 0
    last_result_file: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage64ExternalResultStoreService(Stage63JobRetentionListingService):
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
        result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(),
    ) -> None:
        if result_store_config.inline_result_row_limit < 0:
            raise ValueError("inline_result_row_limit must be >= 0")
        if result_store_config.max_result_page_limit < 1:
            raise ValueError("max_result_page_limit must be >= 1")
        self.result_store_config = result_store_config
        self.result_store_metrics = Stage64ResultStoreMetrics()
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
            job_config=job_config,
            persistence_config=persistence_config,
            retention_config=retention_config,
        )

    def _result_dir(self) -> Path:
        return Path(self.result_store_config.result_dir)

    def _result_path(self, job_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in job_id)
        return self._result_dir() / f"{safe}.jsonl"

    def _write_result_file(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        rows = result.get("rows") or []
        if not isinstance(rows, list):
            rows = []
        result_dir = self._result_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        path = self._result_path(job_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        bytes_written = path.stat().st_size
        self.result_store_metrics.result_files_written += 1
        self.result_store_metrics.result_rows_written += len(rows)
        self.result_store_metrics.result_bytes_written += bytes_written
        self.result_store_metrics.last_result_file = str(path)
        inline_limit = self.result_store_config.inline_result_row_limit
        return {
            "status": result.get("status", "ok"),
            "row_count": len(rows),
            "rows_inline": rows[:inline_limit],
            "rows_inline_count": min(len(rows), inline_limit),
            "result_ref": {
                "storage": "jsonl",
                "path": str(path),
                "bytes": bytes_written,
            },
            "metrics": result.get("metrics"),
        }

    def _serialize_record(self, record: Stage61JobRecord) -> dict[str, Any]:
        value = super()._serialize_record(record)
        result = value.get("result")
        if isinstance(result, dict) and "rows" in result:
            compact = dict(result)
            rows = compact.pop("rows", [])
            compact["row_count"] = len(rows) if isinstance(rows, list) else 0
            compact["rows_inline"] = []
            compact["rows_inline_count"] = 0
            compact.setdefault("result_ref", None)
            value["result"] = compact
        return value

    def _delete_result_for_job(self, job_id: str) -> bool:
        path = self._result_path(job_id)
        if path.exists():
            path.unlink()
            self.result_store_metrics.result_files_deleted += 1
            return True
        return False

    def _apply_retention_locked(self, now: float) -> list[str]:
        removed = super()._apply_retention_locked(now)
        if self.result_store_config.delete_result_file_on_job_prune:
            for job_id in removed:
                self._delete_result_for_job(job_id)
        return removed

    def _update_job(self, job_id: str, **updates: Any) -> Stage61JobRecord:
        result = updates.get("result")
        if isinstance(result, dict) and "rows" in result:
            updates = dict(updates)
            updates["result"] = self._write_result_file(job_id, result)
        return super()._update_job(job_id, **updates)

    def _public_job(self, record: Stage61JobRecord, *, include_payload: bool = False, include_result_rows: bool = False) -> dict[str, Any]:
        value = record.public(include_payload=include_payload, security=self.security)
        result = value.get("result")
        if isinstance(result, dict) and not include_result_rows:
            compact = dict(result)
            compact.pop("rows", None)
            value["result"] = compact
        return value

    def read_result_page(self, job_id: str, *, offset: int = 0, limit: int = 50) -> dict[str, Any] | None:
        offset = max(0, offset)
        limit = max(1, min(limit, self.result_store_config.max_result_page_limit))
        record = self.get_job(job_id)
        if record is None:
            return None
        result = record.result or {}
        result_ref = result.get("result_ref") if isinstance(result, dict) else None
        path = Path(result_ref.get("path")) if isinstance(result_ref, dict) and result_ref.get("path") else self._result_path(job_id)
        if not path.exists():
            return {
                "job_id": job_id,
                "status": record.status,
                "offset": offset,
                "limit": limit,
                "total_rows": int(result.get("row_count", 0)) if isinstance(result, dict) else 0,
                "rows": [],
                "result_available": False,
            }
        rows: list[dict[str, Any]] = []
        total = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if offset <= total < offset + limit:
                    rows.append(json.loads(line))
                total += 1
        return {
            "job_id": job_id,
            "status": record.status,
            "offset": offset,
            "limit": limit,
            "total_rows": total,
            "rows": rows,
            "result_available": True,
            "result_ref": result_ref,
        }

    def result_store_status(self) -> dict[str, Any]:
        return {
            "config": asdict(self.result_store_config),
            "metrics": self.result_store_metrics.snapshot(),
        }

    def job_status(self) -> dict[str, Any]:
        payload = super().job_status()
        payload["result_store"] = self.result_store_status()
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage64_external_result_store"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage64Handler(base_handler):
            server_version = "Stage64ExternalResultQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path.startswith("/v1/jobs/") and parsed.path.endswith("/result"):
                    if not self._check_access():
                        return
                    parts = parsed.path.strip("/").split("/")
                    if len(parts) != 4:
                        self._send_json(404, {"status": "error", "error": "not found"})
                        return
                    job_id = parts[2]
                    query = parse_qs(parsed.query)
                    try:
                        limit = int(query.get("limit", ["50"])[0])
                        offset = int(query.get("offset", ["0"])[0])
                    except ValueError:
                        self._send_json(400, {"status": "error", "error": "limit and offset must be integers"})
                        return
                    page = service.read_result_page(job_id, offset=offset, limit=limit)
                    if page is None:
                        self._send_json(404, {"status": "error", "error": "job not found"})
                        return
                    self._send_json(200, {"status": "ok", "result": page})
                    return
                if parsed.path.startswith("/v1/jobs/") and not parsed.path.endswith("/result"):
                    if not self._check_access():
                        return
                    job_id = parsed.path.rsplit("/", 1)[-1]
                    record = service.get_job(job_id)
                    if record is None:
                        self._send_json(404, {"status": "error", "error": "job not found"})
                        return
                    query = parse_qs(parsed.query)
                    include_payload = query.get("include_payload", ["true"])[0].lower() in {"1", "true", "yes"}
                    include_result_rows = query.get("include_result_rows", ["false"])[0].lower() in {"1", "true", "yes"}
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "job": service._public_job(
                                record,
                                include_payload=include_payload,
                                include_result_rows=include_result_rows,
                            ),
                        },
                    )
                    return
                super().do_GET()

        return Stage64Handler


def build_stage64_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage64ExternalResultStoreService:
    return Stage64ExternalResultStoreService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage64-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
    )


def build_stage64_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B",
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage64ExternalResultStoreService:
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
    return Stage64ExternalResultStoreService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
    )


def run_stage64_external_result_store_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    service = build_stage64_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=4, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=4, max_list_limit=10),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        _post_json(f"{base}/v1/jobs", {"request": _payload("stage64-job")})
        completed = wait_for_job(base, "stage64-job")
        job_summary = _get_json(f"{base}/v1/jobs/stage64-job")
        result_page = _get_json(f"{base}/v1/jobs/stage64-job/result?limit=1")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    state_text = state_path.read_text(encoding="utf-8") if state_path.exists() else ""
    stage_gates = {
        "job_completed": completed.get("job", {}).get("status") == "completed",
        "state_file_written": state_path.exists(),
        "result_file_written": any(result_dir.glob("*.jsonl")),
        "state_does_not_inline_rows": '"rows":' not in state_text,
        "job_summary_has_result_ref": bool(job_summary.get("job", {}).get("result", {}).get("result_ref")),
        "job_summary_omits_rows": "rows" not in (job_summary.get("job", {}).get("result", {}) or {}),
        "result_page_available": result_page.get("result", {}).get("result_available") is True,
        "result_page_limit_applied": len(result_page.get("result", {}).get("rows", [])) <= 1,
        "result_store_visible": "result_store" in admin_jobs.get("jobs", {}),
    }
    summary = {
        "stage": "stage64_external_result_store_smoke",
        "completed": completed,
        "job_summary": job_summary,
        "result_page": result_page,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
