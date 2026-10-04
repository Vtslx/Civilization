from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor, Stage51VersionedFakeRuntime
from .stage59_access_control_audit import (
    Stage59AccessControlledService,
    Stage59SecurityConfig,
    build_stage59_real_service,
    redact_payload,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage60_queue_rate_limit")


@dataclass(frozen=True)
class Stage60QueueConfig:
    max_in_flight: int = 2
    acquire_timeout_seconds: float = 0.1
    per_client_qps: int = 4
    qps_window_seconds: float = 1.0
    max_request_seconds: float = 120.0
    overload_status_code: int = 429


@dataclass
class Stage60QueueMetrics:
    accepted_requests: int = 0
    rejected_overload: int = 0
    rejected_rate_limit: int = 0
    timed_out: int = 0
    completed_requests: int = 0
    in_flight: int = 0
    max_observed_in_flight: int = 0
    last_rejection_reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage60QueueRateLimitedService(Stage59AccessControlledService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
        queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    ) -> None:
        if queue_config.max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if queue_config.per_client_qps < 1:
            raise ValueError("per_client_qps must be >= 1")
        super().__init__(runtime_factory=runtime_factory, descriptor=descriptor, config=config, security=security)
        self.queue_config = queue_config
        self.queue_metrics = Stage60QueueMetrics()
        self._queue_semaphore = threading.BoundedSemaphore(queue_config.max_in_flight)
        self._rate_lock = threading.Lock()
        self._request_times: dict[str, deque[float]] = defaultdict(deque)

    def _rate_allowed(self, client_host: str, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        with self._rate_lock:
            bucket = self._request_times[client_host]
            while bucket and current - bucket[0] > self.queue_config.qps_window_seconds:
                bucket.popleft()
            if len(bucket) >= self.queue_config.per_client_qps:
                self.queue_metrics.rejected_rate_limit += 1
                self.queue_metrics.last_rejection_reason = "rate_limit"
                return False
            bucket.append(current)
            return True

    def _enter_request(self) -> tuple[bool, str | None, float]:
        started = time.perf_counter()
        acquired = self._queue_semaphore.acquire(timeout=self.queue_config.acquire_timeout_seconds)
        if not acquired:
            self.queue_metrics.rejected_overload += 1
            self.queue_metrics.last_rejection_reason = "overload"
            return False, "overload", started
        self.queue_metrics.accepted_requests += 1
        self.queue_metrics.in_flight += 1
        self.queue_metrics.max_observed_in_flight = max(
            self.queue_metrics.max_observed_in_flight,
            self.queue_metrics.in_flight,
        )
        return True, None, started

    def _exit_request(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > self.queue_config.max_request_seconds:
            self.queue_metrics.timed_out += 1
        self.queue_metrics.completed_requests += 1
        self.queue_metrics.in_flight = max(0, self.queue_metrics.in_flight - 1)
        self._queue_semaphore.release()

    def queue_status(self) -> dict[str, Any]:
        return {"config": asdict(self.queue_config), "metrics": self.queue_metrics.snapshot()}

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage60_queue_rate_limit"
        payload["queue"] = self.queue_status()
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage60Handler(base_handler):
            server_version = "Stage60QueueRateLimitedQwenService/1.0"

            def _send_overload(self, reason: str, payload: Any | None = None) -> None:
                status = HTTPStatus.TOO_MANY_REQUESTS if service.queue_config.overload_status_code == 429 else HTTPStatus.SERVICE_UNAVAILABLE
                service._audit(
                    {
                        "event": "queue_rejected",
                        "path": self.path,
                        "method": self.command,
                        "client_host": self.client_address[0],
                        "reason": reason,
                        "payload": redact_payload(payload, config=service.security) if payload is not None else None,
                    }
                )
                self._send_json(
                    status,
                    {
                        "status": "error",
                        "error_type": "RateLimited" if reason == "rate_limit" else "Overloaded",
                        "error": reason,
                        "queue": service.queue_status(),
                    },
                )

            def _queue_admit(self, payload: Any | None = None) -> tuple[bool, float | None]:
                if not service._rate_allowed(self.client_address[0]):
                    self._send_overload("rate_limit", payload)
                    return False, None
                ok, reason, started = service._enter_request()
                if not ok:
                    self._send_overload(reason or "overload", payload)
                    return False, None
                return True, started

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/admin/queue":
                    if not self._check_access():
                        return
                    self._send_json(HTTPStatus.OK, {"status": "ok", "queue": service.queue_status()})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                if self.path in {"/v1/predict", "/v1/batch"}:
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
                    admitted, started = self._queue_admit(payload)
                    if not admitted or started is None:
                        return
                    try:
                        if self.path == "/v1/predict":
                            rows = service.predict_record(payload)
                            self._send_json(HTTPStatus.OK, {"status": "ok", "rows": rows, "metrics": service.metrics.snapshot(), "queue": service.queue_status()})
                            return
                        records = payload.get("requests") if isinstance(payload, dict) else None
                        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                            raise ValueError("batch payload must contain a requests object array")
                        response = service.predict_batch(records)
                        response["queue"] = service.queue_status()
                        self._send_json(HTTPStatus.OK, response)
                        return
                    except Exception as error:
                        service.metrics.failures += 1
                        service.metrics.last_error = str(error)
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                        return
                    finally:
                        service._exit_request(started)
                super().do_POST()

        return Stage60Handler


class Stage60SlowFakeRuntime(Stage51VersionedFakeRuntime):
    def __init__(self, delay_seconds: float = 0.0) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds

    def predict(self, request):  # type: ignore[override]
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return super().predict(request)


def build_stage60_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage60QueueRateLimitedService:
    return Stage60QueueRateLimitedService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage60-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
    )


def build_stage60_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage60QueueRateLimitedService:
    base = build_stage59_real_service(
        port=port,
        security=security,
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
    )
    return Stage60QueueRateLimitedService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
    )


def _post_json(url: str, payload: dict[str, Any] | None = None, *, token: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local test/smoke client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, token: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local test/smoke client
        return json.loads(response.read().decode("utf-8"))


def _payload(index: int = 0) -> dict[str, Any]:
    return {
        "id": f"stage60-{index}",
        "text": "Queue and rate limit test.",
        "memory_items": ["memory"],
        "rule_items": ["rule"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def run_stage60_queue_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    security = Stage59SecurityConfig(audit_log_path=str(output / "audit.jsonl"))
    queue_config = Stage60QueueConfig(max_in_flight=1, acquire_timeout_seconds=0.05, per_client_qps=2)
    service = build_stage60_fake_service(
        port=0,
        security=security,
        queue_config=queue_config,
        runtime_delay_seconds=0.2,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    results: list[dict[str, Any]] = []

    def worker(index: int) -> None:
        try:
            results.append({"index": index, "payload": _post_json(f"{base}/v1/predict", _payload(index), timeout=5.0)})
        except HTTPError as error:
            results.append({"index": index, "error_code": error.code, "error": error.read().decode("utf-8")})

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    try:
        health = _get_json(f"{base}/health")
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        queue_status = _get_json(f"{base}/admin/queue")
    finally:
        service.shutdown()
    status_codes = [item.get("error_code", 200) for item in results]
    stage_gates = {
        "health_ready": health.get("ready") is True,
        "some_success": any(code == 200 for code in status_codes),
        "rate_or_overload_rejected": any(code in {429, 503} for code in status_codes),
        "queue_metrics_recorded": queue_status.get("queue", {}).get("metrics", {}).get("accepted_requests", 0) >= 1,
        "max_in_flight_respected": queue_status.get("queue", {}).get("metrics", {}).get("max_observed_in_flight", 0) <= 1,
    }
    summary = {
        "stage": "stage60_queue_rate_limit_smoke",
        "health": health,
        "results": results,
        "queue_status": queue_status,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
