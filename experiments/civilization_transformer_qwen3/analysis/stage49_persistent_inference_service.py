from __future__ import annotations

from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage45_adapter_package import Stage45InferenceRequest, Stage45InferenceResponse
from .stage48_production_jsonl_inference import (
    Stage48Limits,
    RuntimePredictor,
    _control_trace_valid,
    build_stage48_runtime,
    parse_stage48_record,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage49_persistent_inference_service")
DEFAULT_PACKAGE_MANIFEST = Path("experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json")
DEFAULT_CENTROID_BUNDLE = Path(
    "experiments/civilization_transformer_qwen3/artifacts/stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
)


@dataclass(frozen=True)
class Stage49ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    max_batch_size: int = 32
    batch_window_ms: int = 0
    request_timeout_seconds: float = 120.0
    default_controls: tuple[str, ...] = ("full",)
    limits: Stage48Limits = Stage48Limits()
    capabilities: dict[str, Any] | None = None


@dataclass
class Stage49ServiceMetrics:
    started_at: float = field(default_factory=time.time)
    runtime_loaded_at: float | None = None
    requests_received: int = 0
    prediction_rows: int = 0
    successes: int = 0
    failures: int = 0
    batches: int = 0
    control_audit_failures: int = 0
    last_error: str | None = None
    max_latency_seconds: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "uptime_seconds": time.time() - self.started_at,
            "ready": self.runtime_loaded_at is not None,
        }


class RuntimeFactory(Protocol):
    def __call__(self) -> RuntimePredictor:
        ...


class Stage49PersistentInferenceService:
    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory
        self._runtime: RuntimePredictor | None = None
        self._runtime_lock = threading.RLock()
        self._predict_lock = threading.Lock()
        self._metrics = Stage49ServiceMetrics()
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def metrics(self) -> Stage49ServiceMetrics:
        return self._metrics

    def load_runtime(self) -> None:
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory()
                self._metrics.runtime_loaded_at = time.time()

    def ready(self) -> bool:
        return self._runtime is not None

    def health(self) -> dict[str, Any]:
        return {
            "stage": "stage49_persistent_inference_service",
            "status": "ok",
            "ready": self.ready(),
            "metrics": self._metrics.snapshot(),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "max_batch_size": self.config.max_batch_size,
                "batch_window_ms": self.config.batch_window_ms,
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "default_controls": list(self.config.default_controls),
                "limits": asdict(self.config.limits),
            },
        }

    def _runtime_required(self) -> RuntimePredictor:
        self.load_runtime()
        assert self._runtime is not None
        return self._runtime

    def predict_record(self, payload: dict[str, Any], *, line_number: int = 1) -> list[dict[str, Any]]:
        started = time.perf_counter()
        self._metrics.requests_received += 1
        rows: list[dict[str, Any]] = []
        try:
            request_id, base_request, controls = parse_stage48_record(
                payload,
                limits=self.config.limits,
                default_controls=self.config.default_controls,
            )
        except Exception as error:
            self._metrics.failures += 1
            self._metrics.last_error = str(error)
            return [
                {
                    "id": str(payload.get("id", "")) if isinstance(payload, dict) else "",
                    "line_number": line_number,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ]
        runtime = self._runtime_required()
        if self.config.batch_window_ms > 0:
            time.sleep(self.config.batch_window_ms / 1000.0)
        for control in controls:
            request = Stage45InferenceRequest(**{**asdict(base_request), "control_mode": control})
            try:
                with self._predict_lock:
                    response = runtime.predict(request)
                audit_result = _control_trace_valid(response, control)
                if audit_result is False:
                    self._metrics.control_audit_failures += 1
                self._metrics.successes += response.status == "ok"
                self._metrics.failures += response.status != "ok"
                rows.append(
                    {
                        "id": request_id,
                        "line_number": line_number,
                        "control_mode": control,
                        "status": response.status,
                        "response": asdict(response),
                        "control_audit_passed": audit_result,
                    }
                )
            except Exception as error:
                self._metrics.failures += 1
                self._metrics.last_error = str(error)
                rows.append(
                    {
                        "id": request_id,
                        "line_number": line_number,
                        "control_mode": control,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        elapsed = time.perf_counter() - started
        self._metrics.max_latency_seconds = max(self._metrics.max_latency_seconds, elapsed)
        self._metrics.prediction_rows += len(rows)
        return rows

    def predict_batch(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        if len(payloads) > self.config.max_batch_size:
            raise ValueError("batch exceeds max_batch_size")
        self._metrics.batches += 1
        rows: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads, start=1):
            rows.extend(self.predict_record(payload, line_number=index))
        return {
            "stage": "stage49_persistent_inference_service",
            "status": "ok",
            "rows": rows,
            "metrics": self._metrics.snapshot(),
            "control_audit_passed": all(row.get("control_audit_passed", True) is not False for row in rows),
        }

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        service = self

        class Stage49Handler(BaseHTTPRequestHandler):
            server_version = "Stage49QwenService/1.0"

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Any:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("request body is empty")
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path in {"/health", "/ready"}:
                    payload = service.health()
                    if self.path == "/ready" and not service.ready():
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, payload)
                    else:
                        self._send_json(HTTPStatus.OK, payload)
                    return
                if self.path == "/metrics":
                    self._send_json(HTTPStatus.OK, service.metrics.snapshot())
                    return
                if self.path == "/v1/capabilities":
                    descriptor = getattr(service, "descriptor", None)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "runtime_label": getattr(descriptor, "label", None),
                            "capabilities": service.config.capabilities,
                        },
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                try:
                    payload = self._read_json()
                    if self.path == "/v1/predict":
                        rows = service.predict_record(payload)
                        self._send_json(HTTPStatus.OK, {"status": "ok", "rows": rows, "metrics": service.metrics.snapshot()})
                        return
                    if self.path == "/v1/batch":
                        records = payload.get("requests") if isinstance(payload, dict) else None
                        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                            raise ValueError("batch payload must contain a requests object array")
                        self._send_json(HTTPStatus.OK, service.predict_batch(records))
                        return
                    self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})
                except Exception as error:
                    service.metrics.failures += 1
                    service.metrics.last_error = str(error)
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                    )

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
                return

        return Stage49Handler

    def serve_forever(self) -> None:
        self.load_runtime()
        self._httpd = ThreadingHTTPServer((self.config.host, self.config.port), self.make_handler())
        self._httpd.serve_forever()

    def start_background(self) -> ThreadingHTTPServer:
        self.load_runtime()
        self._httpd = ThreadingHTTPServer((self.config.host, self.config.port), self.make_handler())
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        return self._httpd

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


class Stage49FakeRuntime:
    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        if request.text == "runtime failure":
            raise RuntimeError("synthetic runtime failure")
        trace = {
            "control_mode": request.control_mode,
            "qwen_trainable_parameters": 0,
            "qwen_gradients": 0,
            "weights_unchanged": True,
            "traces": {
                "16": {
                    "memory_delta_norm": 0.0 if request.control_mode in {"no_memory_path", "adapter_disabled", "zero_scale"} else 1.0,
                    "rule_delta_norm": 0.0 if request.control_mode in {"no_rule_path", "adapter_disabled", "zero_scale"} else 1.0,
                    "state_delta_norm": 0.0 if request.control_mode in {"no_state_path", "adapter_disabled", "zero_scale"} else 1.0,
                    "delta_norm": 0.0 if request.control_mode in {"adapter_disabled", "zero_scale"} else 1.0,
                    "hidden_norm_ratio": 1.0,
                }
            },
        }
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=0,
            scores={"projected_delta": [1.0, 0.0], "raw_full_hidden_fixed_centroid": [0.9, 0.1]},
            trace=trace,
        )


def build_stage49_real_service(
    *,
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    centroid_bundle: str | Path = DEFAULT_CENTROID_BUNDLE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preferred_device: str = "cuda",
    max_length: int = 384,
    config: Stage49ServiceConfig = Stage49ServiceConfig(),
) -> Stage49PersistentInferenceService:
    return Stage49PersistentInferenceService(
        runtime_factory=lambda: build_stage48_runtime(
            package_manifest=package_manifest,
            centroid_bundle=centroid_bundle,
            model_path=model_path,
            preferred_device=preferred_device,
            max_length=max_length,
        ),
        config=config,
    )


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def run_stage49_service_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR, port: int = 0) -> dict[str, Any]:
    output = Path(output_dir)
    service = Stage49PersistentInferenceService(
        runtime_factory=Stage49FakeRuntime,
        config=Stage49ServiceConfig(port=port, default_controls=("full", "no_memory_path")),
    )
    server = service.start_background()
    host, actual_port = server.server_address
    base_url = f"http://{host}:{actual_port}"
    try:
        health = _get_json(f"{base_url}/health")
        record = {
            "id": "stage49-smoke-1",
            "text": "Decide the operation using provided memory.",
            "memory_items": ["The memory evidence supports option A."],
            "rule_items": ["Use memory unless a rule overrides it."],
            "state_values": [1.0, 0.0, 0.5],
            "answer_options": ["option A", "option B"],
            "task_name": "operation_decision",
            "seed": 202,
            "controls": ["full", "no_memory_path", "adapter_disabled"],
        }
        predict = _post_json(f"{base_url}/v1/predict", record)
        batch = _post_json(f"{base_url}/v1/batch", {"requests": [record, {"id": "bad", "text": "", "answer_options": []}]})
        metrics = _get_json(f"{base_url}/metrics")
    finally:
        service.shutdown()
    summary = {
        "stage": "stage49_persistent_inference_service_smoke",
        "health": health,
        "predict": predict,
        "batch": batch,
        "metrics": metrics,
        "stage_gates": {
            "service_ready": health.get("ready") is True,
            "predict_ok": predict.get("status") == "ok" and len(predict.get("rows", [])) == 3,
            "batch_isolated_failure": any(row.get("status") == "error" for row in batch.get("rows", [])),
            "control_audit_passed": predict.get("rows", [{}])[0].get("control_audit_passed") is True
            and predict.get("control_audit_passed", True) is not False,
            "metrics_recorded": metrics.get("requests_received", 0) >= 2,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
