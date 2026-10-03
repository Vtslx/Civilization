from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig, redact_payload
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import (
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE61_OUTPUT_DIR,
    Stage61AsyncJobService,
    Stage61JobConfig,
    Stage61JobMetrics,
    Stage61JobRecord,
    _delete_json,
    _get_json,
    _payload,
    _post_json,
    wait_for_job,
)
from experiments.civilization_transformer_qwen3.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage62_persistent_async_jobs")
DEFAULT_JOB_STATE = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage62_jobs_state.json")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage62_jobs.jsonl")


@dataclass(frozen=True)
class Stage62PersistenceConfig:
    job_state_path: str = str(DEFAULT_JOB_STATE)
    recover_incomplete_jobs: bool = True


class Stage62PersistentAsyncJobService(Stage61AsyncJobService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
        queue_config: Stage60QueueConfig = Stage60QueueConfig(),
        job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
        persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(),
    ) -> None:
        self.persistence_config = persistence_config
        self._restored_job_count = 0
        self._requeued_job_count = 0
        self._persist_lock = threading.RLock()
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
            job_config=job_config,
        )
        self._restore_jobs_from_disk()

    def _state_path(self) -> Path:
        return Path(self.persistence_config.job_state_path)

    def _serialize_record(self, record: Stage61JobRecord) -> dict[str, Any]:
        return asdict(record)

    def _persist_jobs(self) -> None:
        with self._persist_lock:
            path = self._state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._jobs_lock:
                records = [self._serialize_record(record) for record in self._jobs.values()]
            payload = {
                "stage": "stage62_persistent_async_jobs",
                "updated_at": time.time(),
                "jobs": records,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _record_from_dict(self, payload: dict[str, Any]) -> Stage61JobRecord:
        return Stage61JobRecord(
            job_id=str(payload["job_id"]),
            status=str(payload["status"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            payload=dict(payload.get("payload") or {}),
            result=payload.get("result"),
            error=payload.get("error"),
            cancel_requested=bool(payload.get("cancel_requested", False)),
        )

    def _restore_jobs_from_disk(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        jobs = state.get("jobs", [])
        if not isinstance(jobs, list):
            raise ValueError(f"invalid Stage62 job state: jobs must be a list in {path}")
        restored: list[Stage61JobRecord] = []
        for item in jobs:
            if not isinstance(item, dict):
                raise ValueError(f"invalid Stage62 job state item in {path}")
            record = self._record_from_dict(item)
            if record.status == "running":
                if self.persistence_config.recover_incomplete_jobs and not record.cancel_requested:
                    record.status = "queued"
                    record.error = None
                    record.updated_at = time.time()
                else:
                    record.status = "canceled"
                    record.cancel_requested = True
                    record.updated_at = time.time()
            if record.status == "queued" and record.cancel_requested:
                record.status = "canceled"
                record.updated_at = time.time()
            restored.append(record)
        with self._jobs_lock:
            self._jobs = {record.job_id: record for record in restored}
        self._restored_job_count = len(restored)
        self._rebuild_metrics()
        if self.persistence_config.recover_incomplete_jobs:
            self._requeue_restored_jobs()
        self._append_job_event(
            {
                "event": "jobs_restored",
                "restored": self._restored_job_count,
                "requeued": self._requeued_job_count,
                "state_path": str(path),
            }
        )
        self._persist_jobs()

    def _rebuild_metrics(self) -> None:
        metrics = Stage61JobMetrics()
        with self._jobs_lock:
            metrics.submitted = len(self._jobs)
            metrics.accepted = len(self._jobs)
            for record in self._jobs.values():
                if record.status == "completed":
                    metrics.completed += 1
                elif record.status == "failed":
                    metrics.failed += 1
                    metrics.last_error = record.error.get("error") if isinstance(record.error, dict) else metrics.last_error
                elif record.status == "canceled":
                    metrics.canceled += 1
        self.job_metrics = metrics

    def _requeue_restored_jobs(self) -> None:
        with self._jobs_lock:
            queued = [record for record in sorted(self._jobs.values(), key=lambda item: item.created_at) if record.status == "queued"]
        for record in queued:
            try:
                self._job_queue.put_nowait(record.job_id)
                self._requeued_job_count += 1
            except Exception:
                record.status = "failed"
                record.error = {"error_type": "RecoveredQueueOverflow", "error": "restored job queue is full"}
                record.updated_at = time.time()
                self.job_metrics.failed += 1
                self.job_metrics.last_error = record.error["error"]
        self.job_metrics.max_observed_queue_depth = max(self.job_metrics.max_observed_queue_depth, self._job_queue.qsize())

    def _update_job(self, job_id: str, **updates: Any) -> Stage61JobRecord:
        record = super()._update_job(job_id, **updates)
        self._persist_jobs()
        return record

    def submit_job(self, payload: dict[str, Any]) -> Stage61JobRecord:
        record = super().submit_job(payload)
        self._persist_jobs()
        return record

    def cancel_job(self, job_id: str) -> Stage61JobRecord | None:
        record = super().cancel_job(job_id)
        if record is not None:
            self._persist_jobs()
        return record

    def job_status(self) -> dict[str, Any]:
        payload = super().job_status()
        payload["persistence"] = {
            "job_state_path": self.persistence_config.job_state_path,
            "recover_incomplete_jobs": self.persistence_config.recover_incomplete_jobs,
            "restored_job_count": self._restored_job_count,
            "requeued_job_count": self._requeued_job_count,
            "state_file_exists": self._state_path().exists(),
        }
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage62_persistent_async_jobs"
        return payload

    def _append_job_event(self, event: dict[str, Any]) -> None:
        safe_event = dict(event)
        if "payload" in safe_event and isinstance(safe_event["payload"], dict):
            safe_event["payload"] = redact_payload(safe_event["payload"], config=self.security)
        super()._append_job_event(safe_event)

    def shutdown(self) -> None:
        self._persist_jobs()
        super().shutdown()


def build_stage62_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage62PersistentAsyncJobService:
    return Stage62PersistentAsyncJobService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage62-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
    )


def build_stage62_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage62PersistentAsyncJobService:
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
    return Stage62PersistentAsyncJobService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
    )


def run_stage62_persistent_async_jobs_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    job_config = Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=4)
    persistence_config = Stage62PersistenceConfig(job_state_path=str(state_path))

    first = build_stage62_fake_service(
        port=0,
        job_config=job_config,
        persistence_config=persistence_config,
        runtime_delay_seconds=0.05,
    )
    server = first.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        submitted = _post_json(f"{base}/v1/jobs", {"request": _payload("stage62-completed")})
        completed = wait_for_job(base, submitted["job"]["job_id"])
        queued_cancel = _post_json(f"{base}/v1/jobs", {"request": _payload("stage62-canceled")})
        canceled = _delete_json(f"{base}/v1/jobs/{queued_cancel['job']['job_id']}")
    finally:
        first.shutdown()

    second = build_stage62_fake_service(
        port=0,
        job_config=job_config,
        persistence_config=persistence_config,
        runtime_delay_seconds=0.0,
    )
    server = second.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        restored_completed = _get_json(f"{base}/v1/jobs/stage62-completed")
        restored_canceled = _get_json(f"{base}/v1/jobs/stage62-canceled")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        second.shutdown()

    log_text = job_log.read_text(encoding="utf-8") if job_log.exists() else ""
    stage_gates = {
        "job_completed_before_restart": completed.get("job", {}).get("status") == "completed",
        "job_canceled_before_restart": canceled.get("job", {}).get("status") in {"queued", "running", "canceled", "completed"},
        "state_file_written": state_path.exists(),
        "completed_restored": restored_completed.get("job", {}).get("status") == "completed",
        "canceled_restored": restored_canceled.get("job", {}).get("status") in {"canceled", "completed"},
        "admin_persistence_visible": admin_jobs.get("jobs", {}).get("persistence", {}).get("restored_job_count", 0) >= 2,
        "job_log_redacted": "Async job smoke." not in log_text and '"memory"' not in log_text and '"rule"' not in log_text,
    }
    summary = {
        "stage": "stage62_persistent_async_jobs_smoke",
        "completed": completed,
        "restored_completed": restored_completed,
        "restored_canceled": restored_canceled,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
