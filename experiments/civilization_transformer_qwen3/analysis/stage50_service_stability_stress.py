from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import resource
import statistics
import time
from typing import Any
from urllib.request import Request, urlopen

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage49_persistent_inference_service import (
    DEFAULT_CENTROID_BUNDLE,
    DEFAULT_PACKAGE_MANIFEST,
    Stage49FakeRuntime,
    Stage49PersistentInferenceService,
    Stage49ServiceConfig,
    build_stage49_real_service,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage50_service_stability_stress")


@dataclass(frozen=True)
class Stage50StressConfig:
    total_requests: int = 24
    concurrency: int = 4
    batch_size: int = 3
    controls: tuple[str, ...] = ("full", "no_memory_path", "no_rule_path", "adapter_disabled")
    max_batch_size: int = 8
    request_timeout_seconds: float = 180.0
    max_failure_rate: float = 0.0
    max_control_audit_failures: int = 0
    max_rss_growth_mb: float = 512.0
    max_cuda_growth_mb: float = 512.0
    port: int = 0


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _rss_mb() -> float:
    # Linux reports kilobytes, macOS reports bytes. The WSL production path is Linux;
    # this fallback keeps local smoke summaries readable.
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / 1024.0 if raw > 10_000_000 else raw / 1024.0


def _cuda_allocated_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _request_payload(index: int, controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": f"stage50-{index:04d}",
        "text": "A controller must decide whether to approve the operation after reviewing the supplied evidence.",
        "memory_items": [
            "The operational evidence supports approving the action.",
            f"Request specific memory marker {index}.",
        ],
        "rule_items": ["Use grounded memory evidence unless a stronger rule overrides it."],
        "state_values": [0.9, 0.1, 0.8],
        "answer_options": ["Approve the action.", "Reject the action."],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": list(controls),
        "readouts": ["projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"],
    }


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local service client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local service client
        return json.loads(response.read().decode("utf-8"))


def _flatten_result(payload: dict[str, Any], *, request_index: int, latency_seconds: float) -> dict[str, Any]:
    rows = payload.get("rows", [])
    failures = [row for row in rows if row.get("status") != "ok"]
    audit_failures = [row for row in rows if row.get("control_audit_passed") is False]
    return {
        "request_index": request_index,
        "latency_seconds": latency_seconds,
        "row_count": len(rows),
        "failure_count": len(failures),
        "control_audit_failure_count": len(audit_failures),
        "status": payload.get("status", "unknown"),
    }


def _flatten_batch_result(payload: dict[str, Any], *, batch_indices: list[int], latency_seconds: float) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    by_line: dict[int, list[dict[str, Any]]] = {line_number: [] for line_number in range(1, len(batch_indices) + 1)}
    for row in rows:
        line_number = int(row.get("line_number", 0))
        if line_number in by_line:
            by_line[line_number].append(row)
    flattened: list[dict[str, Any]] = []
    for line_number, request_index in enumerate(batch_indices, start=1):
        request_rows = by_line.get(line_number, [])
        failures = [row for row in request_rows if row.get("status") != "ok"]
        audit_failures = [row for row in request_rows if row.get("control_audit_passed") is False]
        flattened.append(
            {
                "request_index": request_index,
                "latency_seconds": latency_seconds / max(1, len(batch_indices)),
                "row_count": len(request_rows),
                "failure_count": len(failures),
                "control_audit_failure_count": len(audit_failures),
                "status": payload.get("status", "unknown") if request_rows else "missing",
            }
        )
    return flattened


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def run_stage50_service_stress(
    *,
    service: Stage49PersistentInferenceService,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage50StressConfig = Stage50StressConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rss_preload = _rss_mb()
    cuda_preload = _cuda_allocated_mb()
    server = service.start_background()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    request_results: list[dict[str, Any]] = []
    try:
        health_before = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
        rss_before = _rss_mb()
        cuda_before = _cuda_allocated_mb()
        batches = [
            list(range(start, min(config.total_requests, start + config.batch_size)))
            for start in range(0, config.total_requests, config.batch_size)
        ]

        def submit_batch(batch_indices: list[int]) -> list[dict[str, Any]]:
            payloads = [_request_payload(index, config.controls) for index in batch_indices]
            started_request = time.perf_counter()
            latency = time.perf_counter() - started_request
            try:
                response = _post_json(
                    f"{base_url}/v1/batch",
                    {"requests": payloads},
                    timeout=config.request_timeout_seconds,
                )
                latency = time.perf_counter() - started_request
                return _flatten_batch_result(response, batch_indices=batch_indices, latency_seconds=latency)
            except Exception as error:
                latency = time.perf_counter() - started_request
                return [
                    {
                        "request_index": index,
                        "latency_seconds": latency / max(1, len(batch_indices)),
                        "row_count": 1,
                        "failure_count": 1,
                        "control_audit_failure_count": 0,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    for index in batch_indices
                ]

        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = [executor.submit(submit_batch, batch) for batch in batches]
            for future in as_completed(futures):
                request_results.extend(future.result())
        health_after = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
    finally:
        service.shutdown()
    elapsed = time.perf_counter() - started
    rss_after = _rss_mb()
    cuda_after = _cuda_allocated_mb()
    latencies = [float(row["latency_seconds"]) for row in request_results]
    failure_count = sum(int(row["failure_count"]) for row in request_results)
    audit_failures = sum(int(row["control_audit_failure_count"]) for row in request_results)
    success_rows = sum(int(row["row_count"]) - int(row["failure_count"]) for row in request_results)
    total_rows = sum(int(row["row_count"]) for row in request_results)
    failure_rate = failure_count / total_rows if total_rows else 1.0
    cuda_growth = (cuda_after - cuda_before) if cuda_before is not None and cuda_after is not None else None
    resource_usage = {
        "rss_preload_mb": rss_preload,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_growth_mb": rss_after - rss_before,
        "cuda_preload_mb": cuda_preload,
        "cuda_before_mb": cuda_before,
        "cuda_after_mb": cuda_after,
        "cuda_growth_mb": cuda_growth,
    }
    latency_metrics = {
        "count": len(latencies),
        "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "max_seconds": max(latencies) if latencies else 0.0,
        "p50_seconds": _percentile(latencies, 50),
        "p95_seconds": _percentile(latencies, 95),
    }
    stage_gates = {
        "service_ready_before": health_before.get("ready") is True,
        "service_ready_after": health_after.get("ready") is True,
        "all_requests_completed": len(request_results) == config.total_requests,
        "failure_rate": failure_rate <= config.max_failure_rate,
        "control_audit_failures": audit_failures <= config.max_control_audit_failures,
        "rss_growth": resource_usage["rss_growth_mb"] <= config.max_rss_growth_mb,
        "cuda_growth": cuda_growth is None or cuda_growth <= config.max_cuda_growth_mb,
    }
    summary = {
        "stage": "stage50_service_stability_stress",
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "health_before": health_before,
        "health_after": health_after,
        "total_prediction_rows": total_rows,
        "success_rows": success_rows,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "control_audit_failures": audit_failures,
        "latency_metrics": latency_metrics,
        "resource_usage": resource_usage,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "resource_usage.json", resource_usage)
    with (output / "request_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(request_results, key=lambda item: int(item["request_index"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "latency_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "request_index",
                "latency_seconds",
                "row_count",
                "failure_count",
                "control_audit_failure_count",
                "status",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in sorted(request_results, key=lambda item: int(item["request_index"])):
            writer.writerow(row)
    return summary


def run_stage50_fake_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage50StressConfig = Stage50StressConfig(),
) -> dict[str, Any]:
    service = Stage49PersistentInferenceService(
        runtime_factory=Stage49FakeRuntime,
        config=Stage49ServiceConfig(port=config.port, max_batch_size=config.max_batch_size),
    )
    return run_stage50_service_stress(service=service, output_dir=output_dir, config=config)


def run_stage50_real_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage50StressConfig = Stage50StressConfig(total_requests=6, concurrency=2, batch_size=1),
    package_manifest: str | Path = DEFAULT_PACKAGE_MANIFEST,
    centroid_bundle: str | Path = DEFAULT_CENTROID_BUNDLE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> dict[str, Any]:
    service = build_stage49_real_service(
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
        config=Stage49ServiceConfig(port=config.port, max_batch_size=config.max_batch_size),
    )
    return run_stage50_service_stress(service=service, output_dir=output_dir, config=config)
