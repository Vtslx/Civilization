from __future__ import annotations

from dataclasses import asdict, dataclass, field
from http import HTTPStatus
import json
from pathlib import Path
import queue
import threading
import time
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig, redact_payload
from .stage60_queue_rate_limit import (
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE60_OUTPUT_DIR,
    Stage60QueueConfig,
    Stage60QueueRateLimitedService,
    Stage60SlowFakeRuntime,
    build_stage60_real_service,
)
from .stage49_persistent_inference_service import Stage49ServiceConfig


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage61_async_jobs")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage61_jobs.jsonl")


@dataclass(frozen=True)
class Stage61JobConfig:
    max_queued_jobs: int = 128
    worker_count: int = 1
    job_log_path: str = str(DEFAULT_JOB_LOG)
    result_ttl_seconds: float = 3600.0


@dataclass
class Stage61JobRecord:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False

    def public(self, *, include_payload: bool = False, security: Stage59SecurityConfig | None = None) -> dict[str, Any]:
        value = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }
        if include_payload:
            value["payload"] = redact_payload(self.payload, config=security or Stage59SecurityConfig())
        return value


@dataclass
class Stage61JobMetrics:
    submitted: int = 0
    accepted: int = 0
    rejected_queue_full: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    canceled: int = 0
    max_observed_queue_depth: int = 0
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage61AsyncJobService(Stage60QueueRateLimitedService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
        queue_config: Stage60QueueConfig = Stage60QueueConfig(),
        job_config: Stage61JobConfig = Stage61JobConfig(),
    ) -> None:
        if job_config.max_queued_jobs < 1:
            raise ValueError("max_queued_jobs must be >= 1")
        if job_config.worker_count != 1:
            raise ValueError("Stage61 currently supports exactly one worker")
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
        )
        self.job_config = job_config
        self.job_metrics = Stage61JobMetrics()
        self._jobs: dict[str, Stage61JobRecord] = {}
        self._jobs_lock = threading.RLock()
        self._job_queue: queue.Queue[str] = queue.Queue(maxsize=job_config.max_queued_jobs)
        self._job_stop = threading.Event()
        self._job_worker: threading.Thread | None = None

    def load_runtime(self) -> None:
        super().load_runtime()
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._job_worker is not None and self._job_worker.is_alive():
            return
        self._job_stop.clear()
        self._job_worker = threading.Thread(target=self._worker_loop, name="stage61-job-worker", daemon=True)
        self._job_worker.start()

    def _append_job_event(self, event: dict[str, Any]) -> None:
        path = Path(self.job_config.job_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": time.time(), **event}, ensure_ascii=False) + "\n")

    def _update_job(self, job_id: str, **updates: Any) -> Stage61JobRecord:
        with self._jobs_lock:
            record = self._jobs[job_id]
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = time.time()
            return record

    def submit_job(self, payload: dict[str, Any]) -> Stage61JobRecord:
        self._ensure_worker()
        self.job_metrics.submitted += 1
        if self._job_queue.full():
            self.job_metrics.rejected_queue_full += 1
            raise OverflowError("job queue is full")
        job_id = str(payload.get("id") or uuid.uuid4())
        now = time.time()
        record = Stage61JobRecord(job_id=job_id, status="queued", created_at=now, updated_at=now, payload=payload)
        with self._jobs_lock:
            if job_id in self._jobs:
                raise ValueError(f"duplicate job_id: {job_id}")
            self._jobs[job_id] = record
        self._job_queue.put_nowait(job_id)
        self.job_metrics.accepted += 1
        self.job_metrics.max_observed_queue_depth = max(self.job_metrics.max_observed_queue_depth, self._job_queue.qsize())
        self._append_job_event({"event": "job_submitted", "job_id": job_id, "payload": redact_payload(payload, config=self.security)})
        return record

    def get_job(self, job_id: str) -> Stage61JobRecord | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Stage61JobRecord | None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status == "queued":
                record.status = "canceled"
                record.cancel_requested = True
                record.updated_at = time.time()
                self.job_metrics.canceled += 1
                self._append_job_event({"event": "job_canceled", "job_id": job_id})
            elif record.status == "running":
                record.cancel_requested = True
                record.updated_at = time.time()
                self._append_job_event({"event": "job_cancel_requested", "job_id": job_id})
            return record

    def job_status(self) -> dict[str, Any]:
        with self._jobs_lock:
            counts: dict[str, int] = {}
            for record in self._jobs.values():
                counts[record.status] = counts.get(record.status, 0) + 1
        return {
            "config": asdict(self.job_config),
            "metrics": self.job_metrics.snapshot(),
            "queue_depth": self._job_queue.qsize(),
            "status_counts": counts,
        }

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage61_async_jobs"
        payload["jobs"] = self.job_status()
        return payload

    def _worker_loop(self) -> None:
        while not self._job_stop.is_set():
            try:
                job_id = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            record = self.get_job(job_id)
            if record is None:
                self._job_queue.task_done()
                continue
            if record.status == "canceled":
                self._job_queue.task_done()
                continue
            self._update_job(job_id, status="running")
            self.job_metrics.running += 1
            self._append_job_event({"event": "job_started", "job_id": job_id})
            try:
                if self.get_job(job_id) and self.get_job(job_id).cancel_requested:
                    self._update_job(job_id, status="canceled")
                    self.job_metrics.canceled += 1
                    self._append_job_event({"event": "job_canceled_before_run", "job_id": job_id})
                    continue
                rows = self.predict_record(record.payload)
                result = {"status": "ok", "rows": rows, "metrics": self.metrics.snapshot()}
                self._update_job(job_id, status="completed", result=result)
                self.job_metrics.completed += 1
                self._append_job_event({"event": "job_completed", "job_id": job_id})
            except Exception as error:
                err = {"error_type": type(error).__name__, "error": str(error)}
                self._update_job(job_id, status="failed", error=err)
                self.job_metrics.failed += 1
                self.job_metrics.last_error = str(error)
                self._append_job_event({"event": "job_failed", "job_id": job_id, **err})
            finally:
                self.job_metrics.running = max(0, self.job_metrics.running - 1)
                self._job_queue.task_done()

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage61Handler(base_handler):
            server_version = "Stage61AsyncJobQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/admin/jobs":
                    if not self._check_access():
                        return
                    self._send_json(200, {"status": "ok", "jobs": service.job_status()})
                    return
                if self.path.startswith("/v1/jobs/"):
                    if not self._check_access():
                        return
                    job_id = self.path.rsplit("/", 1)[-1]
                    record = service.get_job(job_id)
                    if record is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "job not found"})
                        return
                    self._send_json(200, {"status": "ok", "job": record.public(include_payload=True, security=service.security)})
                    return
                super().do_GET()

            def do_DELETE(self) -> None:  # noqa: N802 - stdlib API
                if self.path.startswith("/v1/jobs/"):
                    if not self._check_access():
                        return
                    job_id = self.path.rsplit("/", 1)[-1]
                    record = service.cancel_job(job_id)
                    if record is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "job not found"})
                        return
                    self._send_json(200, {"status": "ok", "job": record.public(include_payload=True, security=service.security)})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/v1/jobs":
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                        return
                    if not self._check_access(payload):
                        return
                    request_payload = payload.get("request") if isinstance(payload, dict) and "request" in payload else payload
                    if not isinstance(request_payload, dict):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": "job request must be an object"})
                        return
                    try:
                        record = service.submit_job(request_payload)
                    except OverflowError as error:
                        service._audit(
                            {
                                "event": "job_rejected",
                                "path": self.path,
                                "method": self.command,
                                "client_host": self.client_address[0],
                                "reason": "queue_full",
                            }
                        )
                        self._send_json(
                            HTTPStatus.TOO_MANY_REQUESTS,
                            {"status": "error", "error_type": "JobQueueFull", "error": str(error), "jobs": service.job_status()},
                        )
                        return
                    except Exception as error:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                        return
                    self._send_json(HTTPStatus.ACCEPTED, {"status": "accepted", "job": record.public(include_payload=True, security=service.security)})
                    return
                super().do_POST()

        return Stage61Handler

    def shutdown(self) -> None:
        self._job_stop.set()
        if self._job_worker is not None:
            self._job_worker.join(timeout=2.0)
            self._job_worker = None
        super().shutdown()


def build_stage61_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage61AsyncJobService:
    return Stage61AsyncJobService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage61-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
    )


def build_stage61_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B",
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage61AsyncJobService:
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
    return Stage61AsyncJobService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
    )


def _post_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float = 10.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local test/smoke client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local test/smoke client
        return json.loads(response.read().decode("utf-8"))


def _delete_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    request = Request(url, method="DELETE")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local test/smoke client
        return json.loads(response.read().decode("utf-8"))


def _payload(job_id: str = "stage61-job") -> dict[str, Any]:
    return {
        "id": job_id,
        "text": "Async job smoke.",
        "memory_items": ["memory"],
        "rule_items": ["rule"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def wait_for_job(base_url: str, job_id: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        last = _get_json(f"{base_url}/v1/jobs/{job_id}")
        if last.get("job", {}).get("status") in {"completed", "failed", "canceled"}:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish; last={last}")


def run_stage61_async_jobs_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    service = build_stage61_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(output / "jobs.jsonl"), max_queued_jobs=4),
        runtime_delay_seconds=0.05,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = _get_json(f"{base}/health")
        submitted = _post_json(f"{base}/v1/jobs", {"request": _payload("stage61-smoke")})
        job_id = submitted["job"]["job_id"]
        completed = wait_for_job(base, job_id)
        queued_cancel = _post_json(f"{base}/v1/jobs", {"request": _payload("stage61-cancel")})
        canceled = _delete_json(f"{base}/v1/jobs/{queued_cancel['job']['job_id']}")
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()
    job_log = output / "jobs.jsonl"
    stage_gates = {
        "health_ready": health.get("ready") is True,
        "submit_accepted": submitted.get("status") == "accepted",
        "job_completed": completed.get("job", {}).get("status") == "completed",
        "job_has_result": completed.get("job", {}).get("result", {}).get("status") == "ok",
        "cancel_endpoint_available": canceled.get("job", {}).get("status") in {"queued", "running", "canceled", "completed"},
        "admin_jobs_available": admin_jobs.get("status") == "ok",
        "job_log_written": job_log.exists() and "job_submitted" in job_log.read_text(encoding="utf-8"),
    }
    summary = {
        "stage": "stage61_async_jobs_smoke",
        "health": health,
        "submitted": submitted,
        "completed": completed,
        "canceled": canceled,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
