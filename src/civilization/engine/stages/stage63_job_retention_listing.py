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
from .stage61_async_jobs import Stage61JobConfig, Stage61JobRecord, _delete_json, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import (
    DEFAULT_JOB_LOG as DEFAULT_STAGE62_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE62_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE62_OUTPUT_DIR,
    Stage62PersistenceConfig,
    Stage62PersistentAsyncJobService,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage63_job_retention_listing")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage63_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage63_jobs.jsonl")
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


@dataclass(frozen=True)
class Stage63RetentionConfig:
    max_persisted_jobs: int = 1000
    cleanup_on_persist: bool = True
    max_list_limit: int = 200


@dataclass
class Stage63RetentionMetrics:
    cleanup_runs: int = 0
    pruned_jobs: int = 0
    ttl_pruned_jobs: int = 0
    capacity_pruned_jobs: int = 0
    last_cleanup_at: float | None = None
    last_state_file_bytes: int = 0

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage63JobRetentionListingService(Stage62PersistentAsyncJobService):
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
    ) -> None:
        if retention_config.max_persisted_jobs < 1:
            raise ValueError("max_persisted_jobs must be >= 1")
        if retention_config.max_list_limit < 1:
            raise ValueError("max_list_limit must be >= 1")
        self.retention_config = retention_config
        self.retention_metrics = Stage63RetentionMetrics()
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
            job_config=job_config,
            persistence_config=persistence_config,
        )

    def _apply_retention_locked(self, now: float) -> list[str]:
        removed: list[str] = []
        ttl = self.job_config.result_ttl_seconds
        if ttl >= 0:
            expired = [
                record.job_id
                for record in self._jobs.values()
                if record.status in TERMINAL_STATUSES and now - record.updated_at > ttl
            ]
            for job_id in expired:
                if self._jobs.pop(job_id, None) is not None:
                    removed.append(job_id)
                    self.retention_metrics.ttl_pruned_jobs += 1

        overflow = len(self._jobs) - self.retention_config.max_persisted_jobs
        if overflow > 0:
            terminal = sorted(
                [record for record in self._jobs.values() if record.status in TERMINAL_STATUSES],
                key=lambda record: record.updated_at,
            )
            for record in terminal[:overflow]:
                if self._jobs.pop(record.job_id, None) is not None:
                    removed.append(record.job_id)
                    self.retention_metrics.capacity_pruned_jobs += 1

        if removed:
            self.retention_metrics.pruned_jobs += len(removed)
        self.retention_metrics.cleanup_runs += 1
        self.retention_metrics.last_cleanup_at = now
        return removed

    def cleanup_jobs(self) -> dict[str, Any]:
        with self._jobs_lock:
            before = len(self._jobs)
            removed = self._apply_retention_locked(time.time())
            after = len(self._jobs)
        self._persist_jobs()
        self._append_job_event({"event": "jobs_cleanup", "removed": len(removed), "removed_job_ids": removed})
        return {
            "before": before,
            "after": after,
            "removed": len(removed),
            "removed_job_ids": removed,
            "retention": self.retention_status(),
        }

    def _persist_jobs(self) -> None:
        if getattr(self, "retention_config", None) is not None and self.retention_config.cleanup_on_persist:
            with self._jobs_lock:
                self._apply_retention_locked(time.time())
        super()._persist_jobs()
        state_path = Path(self.persistence_config.job_state_path)
        if state_path.exists():
            self.retention_metrics.last_state_file_bytes = state_path.stat().st_size

    def retention_status(self) -> dict[str, Any]:
        return {
            "config": asdict(self.retention_config),
            "metrics": self.retention_metrics.snapshot(),
        }

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_payload: bool = False,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, self.retention_config.max_list_limit))
        offset = max(0, offset)
        with self._jobs_lock:
            records = list(self._jobs.values())
        if status:
            records = [record for record in records if record.status == status]
        records.sort(key=lambda record: record.created_at, reverse=True)
        selected = records[offset : offset + limit]
        return {
            "total": len(records),
            "offset": offset,
            "limit": limit,
            "status_filter": status,
            "jobs": [
                record.public(include_payload=include_payload, security=self.security)
                for record in selected
            ],
        }

    def job_status(self) -> dict[str, Any]:
        payload = super().job_status()
        payload["retention"] = self.retention_status()
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage63_job_retention_listing"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage63Handler(base_handler):
            server_version = "Stage63JobRetentionQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/v1/jobs":
                    if not self._check_access():
                        return
                    query = parse_qs(parsed.query)
                    status = query.get("status", [None])[0]
                    include_payload = query.get("include_payload", ["false"])[0].lower() in {"1", "true", "yes"}
                    try:
                        limit = int(query.get("limit", ["50"])[0])
                        offset = int(query.get("offset", ["0"])[0])
                    except ValueError:
                        self._send_json(400, {"status": "error", "error": "limit and offset must be integers"})
                        return
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "jobs": service.list_jobs(
                                status=status,
                                limit=limit,
                                offset=offset,
                                include_payload=include_payload,
                            ),
                        },
                    )
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/admin/jobs/cleanup":
                    if not self._check_access():
                        return
                    self._send_json(200, {"status": "ok", "cleanup": service.cleanup_jobs()})
                    return
                super().do_POST()

        return Stage63Handler


def build_stage63_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage63JobRetentionListingService:
    return Stage63JobRetentionListingService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage63-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
    )


def build_stage63_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage63JobRetentionListingService:
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
    return Stage63JobRetentionListingService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
    )


def run_stage63_job_retention_listing_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    service = build_stage63_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=4, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=2, max_list_limit=10),
        runtime_delay_seconds=0.01,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        for index in range(3):
            job_id = f"stage63-job-{index}"
            _post_json(f"{base}/v1/jobs", {"request": _payload(job_id)})
            wait_for_job(base, job_id)
        listed = _get_json(f"{base}/v1/jobs?limit=2")
        filtered = _get_json(f"{base}/v1/jobs?status=completed&limit=10")
        cleanup = _post_json(f"{base}/admin/jobs/cleanup")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    log_text = job_log.read_text(encoding="utf-8") if job_log.exists() else ""
    stage_gates = {
        "listing_available": listed.get("jobs", {}).get("total", 0) >= 2,
        "pagination_limit_applied": len(listed.get("jobs", {}).get("jobs", [])) <= 2,
        "status_filter_available": filtered.get("jobs", {}).get("status_filter") == "completed",
        "cleanup_endpoint_available": cleanup.get("status") == "ok",
        "retention_visible": "retention" in admin_jobs.get("jobs", {}),
        "capacity_pruned": admin_jobs.get("jobs", {}).get("retention", {}).get("metrics", {}).get("capacity_pruned_jobs", 0) >= 1,
        "state_file_written": state_path.exists(),
        "job_log_redacted": "Async job smoke." not in log_text and '"memory"' not in log_text and '"rule"' not in log_text,
    }
    summary = {
        "stage": "stage63_job_retention_listing_smoke",
        "listed": listed,
        "filtered": filtered,
        "cleanup": cleanup,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
