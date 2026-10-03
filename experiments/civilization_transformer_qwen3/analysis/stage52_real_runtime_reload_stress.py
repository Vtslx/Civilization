from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import time
from typing import Any
from urllib.request import Request, urlopen

from .adapter_benchmark import DEFAULT_MODEL_PATH
from ..backend import Qwen3Backend
from .stage45_adapter_package import load_stage45_package_manifest
from .stage46_runtime_inference import Stage46QwenRuntimeBackend
from .stage47_centroid_batch_inference import Stage47CentroidProvider
from .stage49_persistent_inference_service import DEFAULT_CENTROID_BUNDLE, DEFAULT_PACKAGE_MANIFEST, Stage49ServiceConfig
from .stage51_runtime_management import (
    Stage51ManagedInferenceService,
    Stage51RuntimeDescriptor,
    build_stage51_fake_service,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage52_real_runtime_reload_stress")


@dataclass(frozen=True)
class Stage52ReloadStressConfig:
    reload_rounds: int = 1
    controls: tuple[str, ...] = ("full", "no_memory_path", "adapter_disabled")
    request_timeout_seconds: float = 240.0
    max_rss_growth_mb: float = 768.0
    max_cuda_growth_mb: float = 768.0
    port: int = 0


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _cuda_allocated_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _request_payload(controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": "stage52-real-reload",
        "text": "A controller must decide whether to approve the operation after reviewing the supplied evidence.",
        "memory_items": ["The operational evidence supports approving the action."],
        "rule_items": ["Use grounded memory evidence unless a stronger rule overrides it."],
        "state_values": [0.9, 0.1, 0.8],
        "answer_options": ["Approve the action.", "Reject the action."],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": list(controls),
        "readouts": ["projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"],
    }


def _post_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local service client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local service client
        return json.loads(response.read().decode("utf-8"))


def build_stage52_real_managed_service(
    *,
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    centroid_bundle: str | Path = DEFAULT_CENTROID_BUNDLE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preferred_device: str = "cuda",
    max_length: int = 384,
    port: int = 0,
) -> Stage51ManagedInferenceService:
    descriptor = Stage51RuntimeDescriptor(
        package_manifest=str(package_manifest),
        centroid_bundle=str(centroid_bundle),
        model_path=str(model_path),
        preferred_device=preferred_device,
        max_length=max_length,
        label=f"qwen3-{preferred_device}-len{max_length}",
    )
    shared_backend: list[Qwen3Backend | None] = [None]

    def runtime_factory() -> Stage46QwenRuntimeBackend:
        if shared_backend[0] is None:
            shared_backend[0] = Qwen3Backend(model_path, preferred_device=preferred_device)
        manifest = load_stage45_package_manifest(package_manifest)
        provider = Stage47CentroidProvider(centroid_bundle)
        return Stage46QwenRuntimeBackend(
            manifest,
            model_path=model_path,
            preferred_device=preferred_device,
            max_length=max_length,
            raw_centroid_provider=provider,
            backend=shared_backend[0],
        )

    return Stage51ManagedInferenceService(
        runtime_factory=runtime_factory,
        descriptor=descriptor,
        config=Stage49ServiceConfig(port=port),
    )


def _prediction_ok(payload: dict[str, Any]) -> bool:
    rows = payload.get("rows", [])
    return bool(rows) and all(row.get("status") == "ok" and row.get("control_audit_passed") is not False for row in rows)


def run_stage52_reload_stress(
    *,
    service: Stage51ManagedInferenceService,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage52ReloadStressConfig = Stage52ReloadStressConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    server = service.start_background()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        health_before = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
        baseline = _post_json(f"{base_url}/v1/predict", _request_payload(config.controls), timeout=config.request_timeout_seconds)
        rss_before = _rss_mb()
        cuda_before = _cuda_allocated_mb()
        for index in range(config.reload_rounds):
            drain = _post_json(f"{base_url}/admin/drain", timeout=config.request_timeout_seconds)
            draining_predict = _post_json(
                f"{base_url}/v1/predict",
                _request_payload(("full",)),
                timeout=config.request_timeout_seconds,
            )
            reload_response = _post_json(f"{base_url}/admin/reload", timeout=config.request_timeout_seconds)
            resume = _post_json(f"{base_url}/admin/resume", timeout=config.request_timeout_seconds)
            after = _post_json(f"{base_url}/v1/predict", _request_payload(config.controls), timeout=config.request_timeout_seconds)
            events.append(
                {
                    "round": index + 1,
                    "drain": drain,
                    "draining_predict": draining_predict,
                    "reload": reload_response,
                    "resume": resume,
                    "after": after,
                }
            )
        health_after = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
    finally:
        service.shutdown()
    elapsed = time.perf_counter() - started
    rss_after = _rss_mb()
    cuda_after = _cuda_allocated_mb()
    cuda_growth = (cuda_after - cuda_before) if cuda_before is not None and cuda_after is not None else None
    resource_usage = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_growth_mb": rss_after - rss_before,
        "cuda_before_mb": cuda_before,
        "cuda_after_mb": cuda_after,
        "cuda_growth_mb": cuda_growth,
    }
    reload_versions = [
        event["reload"].get("event", {}).get("new_version")
        for event in events
        if event["reload"].get("event", {}).get("status") == "ok"
    ]
    draining_rejections = [
        event["draining_predict"].get("rows", [{}])[0].get("error_type") == "ServiceDraining"
        for event in events
    ]
    post_reload_predictions = [_prediction_ok(event["after"]) for event in events]
    stage_gates = {
        "service_ready_before": health_before.get("ready") is True,
        "service_ready_after": health_after.get("ready") is True,
        "baseline_prediction": _prediction_ok(baseline),
        "reload_count": len(reload_versions) == config.reload_rounds,
        "reload_versions_increase": reload_versions == sorted(reload_versions) and len(set(reload_versions)) == len(reload_versions),
        "drain_rejects_prediction": all(draining_rejections),
        "resume_predictions_ok": all(post_reload_predictions),
        "rss_growth": resource_usage["rss_growth_mb"] <= config.max_rss_growth_mb,
        "cuda_growth": cuda_growth is None or cuda_growth <= config.max_cuda_growth_mb,
    }
    summary = {
        "stage": "stage52_real_runtime_reload_stress",
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "health_before": health_before,
        "health_after": health_after,
        "baseline": baseline,
        "events": events,
        "resource_usage": resource_usage,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "resource_usage.json", resource_usage)
    with (output / "reload_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return summary


def run_stage52_fake_reload_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage52ReloadStressConfig = Stage52ReloadStressConfig(),
) -> dict[str, Any]:
    service = build_stage51_fake_service(port=config.port)
    return run_stage52_reload_stress(service=service, output_dir=output_dir, config=config)


def run_stage52_real_reload_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage52ReloadStressConfig = Stage52ReloadStressConfig(),
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    centroid_bundle: str | Path = DEFAULT_CENTROID_BUNDLE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> dict[str, Any]:
    service = build_stage52_real_managed_service(
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
        port=config.port,
    )
    return run_stage52_reload_stress(service=service, output_dir=output_dir, config=config)
