from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol

from .stage45_adapter_package import (
    INFERENCE_CONTROL_MODES,
    Stage45InferenceRequest,
    Stage45InferenceResponse,
    load_stage45_package_manifest,
)
from .stage46_runtime_inference import Stage46QwenRuntimeBackend
from .stage47_centroid_batch_inference import Stage47CentroidProvider


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage48_jsonl_inference")


@dataclass(frozen=True)
class Stage48Limits:
    max_requests: int = 1000
    max_text_chars: int = 32_000
    max_context_items: int = 64
    max_context_item_chars: int = 4_000
    max_answer_options: int = 16


class RuntimePredictor(Protocol):
    def predict(self, request: Stage45InferenceRequest) -> Stage45InferenceResponse:
        ...


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _tuple_strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON string array")
    return tuple(value)


def _control_trace_valid(response: Stage45InferenceResponse, control: str) -> bool | None:
    traces = response.trace.get("traces")
    if not isinstance(traces, dict) or not traces:
        return None
    if control == "no_memory_path":
        return all(float(trace["memory_delta_norm"]) == 0.0 for trace in traces.values())
    if control == "no_rule_path":
        return all(float(trace["rule_delta_norm"]) == 0.0 for trace in traces.values())
    if control == "no_state_path":
        return all(float(trace["state_delta_norm"]) == 0.0 for trace in traces.values())
    if control in {"adapter_disabled", "zero_scale"}:
        return all(float(trace["delta_norm"]) == 0.0 for trace in traces.values())
    return True


def parse_stage48_record(
    payload: dict[str, Any],
    *,
    limits: Stage48Limits,
    default_controls: tuple[str, ...],
) -> tuple[str, Stage45InferenceRequest, tuple[str, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("JSONL record must be an object")
    request_id = str(payload.get("id", ""))
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > limits.max_text_chars:
        raise ValueError("text exceeds max_text_chars")
    memory_items = _tuple_strings(payload.get("memory_items", []), "memory_items")
    rule_items = _tuple_strings(payload.get("rule_items", []), "rule_items")
    if len(memory_items) + len(rule_items) > limits.max_context_items:
        raise ValueError("context item count exceeds max_context_items")
    if any(len(item) > limits.max_context_item_chars for item in (*memory_items, *rule_items)):
        raise ValueError("context item exceeds max_context_item_chars")
    state = payload.get("state_values", [0.5, 0.5, 0.5])
    if not isinstance(state, list) or len(state) != 3 or any(not isinstance(value, (int, float)) for value in state):
        raise ValueError("state_values must contain three numbers")
    answer_options = _tuple_strings(payload.get("answer_options"), "answer_options")
    if not 2 <= len(answer_options) <= limits.max_answer_options:
        raise ValueError("answer_options count is outside allowed range")
    controls = tuple(payload.get("controls", default_controls))
    if not controls or any(control not in INFERENCE_CONTROL_MODES for control in controls):
        raise ValueError("controls contains an unsupported mode")
    readouts = tuple(payload.get("readouts", ["projected_delta", "raw_full_hidden_fixed_centroid"]))
    request = Stage45InferenceRequest(
        text=text,
        memory_items=memory_items,
        rule_items=rule_items,
        state_values=tuple(float(value) for value in state),
        answer_options=answer_options,
        task_name=str(payload.get("task_name", "custom")),
        seed=int(payload.get("seed", 202)),
        readouts=readouts,
        control_mode="full",
    )
    return request_id, request, controls


def run_stage48_jsonl_inference(
    *,
    runtime: RuntimePredictor,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    default_controls: tuple[str, ...] = ("full",),
    limits: Stage48Limits = Stage48Limits(),
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    input_count = 0
    output_count = 0
    success_count = 0
    failure_count = 0
    failures: list[dict[str, Any]] = []
    control_audits: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as reader, temporary.open("w", encoding="utf-8") as writer:
        for line_number, raw_line in enumerate(reader, start=1):
            if not raw_line.strip():
                continue
            input_count += 1
            if input_count > limits.max_requests:
                raise ValueError("input exceeds max_requests")
            try:
                payload = json.loads(raw_line)
                request_id, base_request, controls = parse_stage48_record(
                    payload, limits=limits, default_controls=default_controls
                )
            except Exception as error:
                row = {
                    "id": "",
                    "line_number": line_number,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_count += 1
                failure_count += 1
                failures.append(row)
                continue
            for control in controls:
                request = Stage45InferenceRequest(**{**asdict(base_request), "control_mode": control})
                try:
                    response = runtime.predict(request)
                    row = {
                        "id": request_id,
                        "line_number": line_number,
                        "control_mode": control,
                        "status": response.status,
                        "response": asdict(response),
                    }
                    audit_result = _control_trace_valid(response, control)
                    if audit_result is not None:
                        control_audits.append(
                            {"id": request_id, "control_mode": control, "passed": audit_result}
                        )
                    success_count += response.status == "ok"
                    failure_count += response.status != "ok"
                except Exception as error:
                    row = {
                        "id": request_id,
                        "line_number": line_number,
                        "control_mode": control,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    failure_count += 1
                    failures.append(row)
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_count += 1
    temporary.replace(destination)
    summary = {
        "stage": "stage48_production_jsonl_inference",
        "input_path": str(source),
        "output_path": str(destination),
        "input_count": input_count,
        "output_count": output_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "elapsed_seconds": time.perf_counter() - started,
        "default_controls": list(default_controls),
        "limits": asdict(limits),
        "failures": failures,
        "control_audits": control_audits,
        "control_audit_passed": all(row["passed"] for row in control_audits),
        "passes_stage_gate": input_count > 0 and success_count > 0 and all(row["passed"] for row in control_audits),
    }
    _json_dump(Path(summary_path), summary)
    return summary


def build_stage48_runtime(
    *,
    package_manifest: str | Path,
    centroid_bundle: str | Path,
    model_path: str | Path,
    preferred_device: str,
    max_length: int,
) -> Stage46QwenRuntimeBackend:
    manifest = load_stage45_package_manifest(package_manifest)
    provider = Stage47CentroidProvider(centroid_bundle)
    return Stage46QwenRuntimeBackend(
        manifest,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
        raw_centroid_provider=provider,
    )
