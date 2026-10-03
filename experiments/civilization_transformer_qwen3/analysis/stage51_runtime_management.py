from __future__ import annotations

from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gc
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

from .stage45_adapter_package import Stage45InferenceRequest, Stage45InferenceResponse
from .stage49_persistent_inference_service import (
    DEFAULT_CENTROID_BUNDLE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE49_OUTPUT_DIR,
    DEFAULT_PACKAGE_MANIFEST,
    Stage49FakeRuntime,
    Stage49PersistentInferenceService,
    Stage49ServiceConfig,
    RuntimeFactory,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage51_runtime_management")


@dataclass(frozen=True)
class Stage51RuntimeDescriptor:
    package_manifest: str
    centroid_bundle: str
    model_path: str = ""
    preferred_device: str = "cuda"
    max_length: int = 384
    label: str = "default"


@dataclass
class Stage51ManagementState:
    runtime_version: int = 0
    runtime_label: str = "unloaded"
    draining: bool = False
    reload_count: int = 0
    reload_failures: int = 0
    last_reload_error: str | None = None
    reload_events: list[dict[str, Any]] = field(default_factory=list)


class Stage51ManagedInferenceService(Stage49PersistentInferenceService):
    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
    ) -> None:
        super().__init__(runtime_factory=runtime_factory, config=config)
        self.descriptor = descriptor
        self.management = Stage51ManagementState(runtime_label=descriptor.label)
        self._management_lock = threading.RLock()

    def load_runtime(self) -> None:
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory()
                self._metrics.runtime_loaded_at = time.time()
                self.management.runtime_version += 1
                self.management.runtime_label = self.descriptor.label

    def set_draining(self, value: bool) -> dict[str, Any]:
        with self._management_lock:
            self.management.draining = value
            return self.management_status()

    def management_status(self) -> dict[str, Any]:
        return {
            "runtime_version": self.management.runtime_version,
            "runtime_label": self.management.runtime_label,
            "draining": self.management.draining,
            "reload_count": self.management.reload_count,
            "reload_failures": self.management.reload_failures,
            "last_reload_error": self.management.last_reload_error,
            "reload_events": list(self.management.reload_events),
            "descriptor": asdict(self.descriptor),
        }

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage51_runtime_management"
        payload["management"] = self.management_status()
        payload["ready"] = payload["ready"] and not self.management.draining
        return payload

    def reload_runtime(
        self,
        *,
        runtime_factory: RuntimeFactory | None = None,
        descriptor: Stage51RuntimeDescriptor | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        old_version = self.management.runtime_version
        old_runtime = self._runtime
        old_descriptor = self.descriptor
        old_factory = self._runtime_factory
        with self._runtime_lock:
            try:
                if runtime_factory is not None:
                    self._runtime_factory = runtime_factory
                if descriptor is not None:
                    self.descriptor = descriptor
                new_runtime = self._runtime_factory()
                self._runtime = new_runtime
                self._metrics.runtime_loaded_at = time.time()
                self.management.runtime_version += 1
                self.management.runtime_label = self.descriptor.label
                self.management.reload_count += 1
                self.management.last_reload_error = None
                event = {
                    "status": "ok",
                    "old_version": old_version,
                    "new_version": self.management.runtime_version,
                    "label": self.management.runtime_label,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                self.management.reload_events.append(event)
                old_runtime = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return event
            except Exception as error:
                self._runtime = old_runtime
                self.descriptor = old_descriptor
                self._runtime_factory = old_factory
                self.management.reload_failures += 1
                self.management.last_reload_error = str(error)
                event = {
                    "status": "error",
                    "old_version": old_version,
                    "new_version": self.management.runtime_version,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                self.management.reload_events.append(event)
                return event

    def predict_record(self, payload: dict[str, Any], *, line_number: int = 1) -> list[dict[str, Any]]:
        if self.management.draining:
            self.metrics.failures += 1
            return [
                {
                    "id": str(payload.get("id", "")) if isinstance(payload, dict) else "",
                    "line_number": line_number,
                    "status": "error",
                    "error_type": "ServiceDraining",
                    "error": "service is draining and not accepting predictions",
                }
            ]
        rows = super().predict_record(payload, line_number=line_number)
        for row in rows:
            row["runtime_version"] = self.management.runtime_version
            row["runtime_label"] = self.management.runtime_label
        return rows

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        service = self
        base_handler = super().make_handler()

        class Stage51Handler(base_handler):
            server_version = "Stage51QwenManagedService/1.0"

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/admin/drain":
                    self._send_json(HTTPStatus.OK, {"status": "ok", "management": service.set_draining(True)})
                    return
                if self.path == "/admin/resume":
                    self._send_json(HTTPStatus.OK, {"status": "ok", "management": service.set_draining(False)})
                    return
                if self.path == "/admin/reload":
                    event = service.reload_runtime()
                    status = HTTPStatus.OK if event["status"] == "ok" else HTTPStatus.INTERNAL_SERVER_ERROR
                    self._send_json(status, {"status": event["status"], "event": event, "management": service.management_status()})
                    return
                super().do_POST()

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/admin/status":
                    self._send_json(HTTPStatus.OK, {"status": "ok", "management": service.management_status()})
                    return
                super().do_GET()

        return Stage51Handler


class Stage51VersionedFakeRuntime:
    _counter = 0

    def __init__(self) -> None:
        type(self)._counter += 1
        self.instance_id = type(self)._counter

    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        trace = {
            "runtime_instance_id": self.instance_id,
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
            scores={"projected_delta": [1.0, 0.0]},
            trace=trace,
        )


def build_stage51_fake_service(*, port: int = 0) -> Stage51ManagedInferenceService:
    return Stage51ManagedInferenceService(
        runtime_factory=Stage51VersionedFakeRuntime,
        descriptor=Stage51RuntimeDescriptor(
            package_manifest=str(DEFAULT_PACKAGE_MANIFEST),
            centroid_bundle=str(DEFAULT_CENTROID_BUNDLE),
            label="fake-runtime",
        ),
        config=Stage49ServiceConfig(port=port),
    )


def _post_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def _request_payload() -> dict[str, Any]:
    return {
        "id": "stage51-smoke-1",
        "text": "Evaluate the operation.",
        "memory_items": ["memory evidence"],
        "rule_items": ["rule evidence"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "adapter_disabled"],
    }


def run_stage51_management_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR, port: int = 0) -> dict[str, Any]:
    output = Path(output_dir)
    service = build_stage51_fake_service(port=port)
    server = service.start_background()
    host, actual_port = server.server_address
    base_url = f"http://{host}:{actual_port}"
    try:
        health_before = _get_json(f"{base_url}/health")
        predict_before = _post_json(f"{base_url}/v1/predict", _request_payload())
        reload_response = _post_json(f"{base_url}/admin/reload")
        predict_after = _post_json(f"{base_url}/v1/predict", _request_payload())
        drain = _post_json(f"{base_url}/admin/drain")
        predict_draining = _post_json(f"{base_url}/v1/predict", _request_payload())
        resume = _post_json(f"{base_url}/admin/resume")
        status = _get_json(f"{base_url}/admin/status")
    finally:
        service.shutdown()
    before_version = predict_before["rows"][0]["runtime_version"]
    after_version = predict_after["rows"][0]["runtime_version"]
    summary = {
        "stage": "stage51_runtime_management_smoke",
        "health_before": health_before,
        "predict_before": predict_before,
        "reload_response": reload_response,
        "predict_after": predict_after,
        "drain": drain,
        "predict_draining": predict_draining,
        "resume": resume,
        "status": status,
        "stage_gates": {
            "ready": health_before.get("ready") is True,
            "reload_increments_version": after_version == before_version + 1,
            "reload_event_recorded": reload_response.get("event", {}).get("status") == "ok",
            "drain_rejects_prediction": predict_draining["rows"][0]["error_type"] == "ServiceDraining",
            "resume_clears_draining": resume.get("management", {}).get("draining") is False,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
