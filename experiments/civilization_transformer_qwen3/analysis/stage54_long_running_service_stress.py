from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time
from typing import Any

from .stage51_runtime_management import build_stage51_fake_service
from .stage52_real_runtime_reload_stress import _cuda_allocated_mb, _rss_mb, build_stage52_real_managed_service
from .stage53_concurrent_reload_stress import _get_json, _post_json, _predict


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage54_long_running_service_stress")


@dataclass(frozen=True)
class Stage54LongRunConfig:
    duration_seconds: float = 1800.0
    request_interval_seconds: float = 5.0
    reload_interval_seconds: float = 300.0
    snapshot_interval_seconds: float = 60.0
    controls: tuple[str, ...] = ("full", "no_memory_path", "adapter_disabled")
    request_timeout_seconds: float = 240.0
    max_failure_rate: float = 0.0
    max_rss_growth_mb: float = 1024.0
    max_cuda_growth_mb: float = 768.0
    min_reload_count: int = 1
    port: int = 0


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_errors(result: dict[str, Any]) -> list[str]:
    return [
        str(row.get("error_type"))
        for row in result.get("payload", {}).get("rows", [])
        if row.get("status") == "error"
    ]


def _row_versions(result: dict[str, Any]) -> list[int]:
    versions: list[int] = []
    for row in result.get("payload", {}).get("rows", []):
        version = row.get("runtime_version")
        if isinstance(version, int):
            versions.append(version)
    return versions


def _snapshot(base_url: str, *, timeout: float) -> dict[str, Any]:
    return {
        "elapsed_seconds": None,
        "health": _get_json(f"{base_url}/health", timeout=timeout),
        "rss_mb": _rss_mb(),
        "cuda_allocated_mb": _cuda_allocated_mb(),
        "timestamp": time.time(),
    }


def run_stage54_long_running_stress(
    *,
    service,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage54LongRunConfig = Stage54LongRunConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    server = service.start_background()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    started = time.perf_counter()
    deadline = started + config.duration_seconds
    next_request = started
    next_reload = started + config.reload_interval_seconds
    next_snapshot = started
    request_index = 0
    results: list[dict[str, Any]] = []
    reload_events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    request_results_path = output / "request_results.jsonl"
    reload_events_path = output / "reload_events.jsonl"
    snapshots_path = output / "metrics_snapshots.jsonl"
    for path in (request_results_path, reload_events_path, snapshots_path):
        path.write_text("", encoding="utf-8")
    try:
        health_before = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
        baseline = _predict(base_url, request_index, config.controls, config.request_timeout_seconds)
        _append_jsonl(request_results_path, baseline)
        request_index += 1
        rss_before = _rss_mb()
        cuda_before = _cuda_allocated_mb()
        _json_dump(
            output / "running.json",
            {
                "stage": "stage54_long_running_service_stress",
                "status": "running",
                "started_at": time.time(),
                "config": asdict(config),
                "health_before": health_before,
            },
        )
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now >= next_snapshot:
                snap = _snapshot(base_url, timeout=config.request_timeout_seconds)
                snap["elapsed_seconds"] = now - started
                snapshots.append(snap)
                _append_jsonl(snapshots_path, snap)
                _json_dump(
                    output / "running.json",
                    {
                        "stage": "stage54_long_running_service_stress",
                        "status": "running",
                        "elapsed_seconds": now - started,
                        "request_count": request_index,
                        "reload_count": len(reload_events),
                        "snapshot_count": len(snapshots),
                        "latest_snapshot": snap,
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage54_progress": "snapshot",
                            "elapsed_seconds": round(now - started, 3),
                            "request_count": request_index,
                            "reload_count": len(reload_events),
                            "rss_mb": snap["rss_mb"],
                            "cuda_allocated_mb": snap["cuda_allocated_mb"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_snapshot += config.snapshot_interval_seconds
            if now >= next_reload:
                drain = _post_json(f"{base_url}/admin/drain", timeout=config.request_timeout_seconds)
                draining_predict = _predict(base_url, request_index, ("full",), config.request_timeout_seconds)
                request_index += 1
                reload_response = _post_json(f"{base_url}/admin/reload", timeout=config.request_timeout_seconds)
                resume = _post_json(f"{base_url}/admin/resume", timeout=config.request_timeout_seconds)
                reload_events.append(
                    {
                        "elapsed_seconds": now - started,
                        "drain": drain,
                        "draining_predict": draining_predict,
                        "reload": reload_response,
                        "resume": resume,
                    }
                )
                _append_jsonl(reload_events_path, reload_events[-1])
                print(
                    json.dumps(
                        {
                            "stage54_progress": "reload",
                            "elapsed_seconds": round(now - started, 3),
                            "reload_count": len(reload_events),
                            "reload_status": reload_response.get("event", {}).get("status"),
                            "new_version": reload_response.get("event", {}).get("new_version"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_reload += config.reload_interval_seconds
            if now >= next_request:
                result = _predict(base_url, request_index, config.controls, config.request_timeout_seconds)
                results.append(result)
                _append_jsonl(request_results_path, result)
                request_index += 1
                next_request += config.request_interval_seconds
            sleep_until = min(next_request, next_reload, next_snapshot, deadline)
            time.sleep(max(0.0, min(0.25, sleep_until - time.perf_counter())))
        health_after = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
    finally:
        service.shutdown()
    elapsed = time.perf_counter() - started
    rss_after = _rss_mb()
    cuda_after = _cuda_allocated_mb()
    all_results = [baseline, *results, *[event["draining_predict"] for event in reload_events]]
    total_rows = sum(len(result.get("payload", {}).get("rows", [])) for result in all_results)
    error_types = [error for result in all_results for error in _row_errors(result)]
    non_draining_errors = [error for error in error_types if error != "ServiceDraining"]
    successful_versions = [version for result in all_results for version in _row_versions(result)]
    reload_versions = [
        event["reload"].get("event", {}).get("new_version")
        for event in reload_events
        if event["reload"].get("event", {}).get("status") == "ok"
    ]
    known_versions = set(reload_versions)
    initial_version = health_before.get("management", {}).get("runtime_version")
    if isinstance(initial_version, int):
        known_versions.add(initial_version)
    invalid_versions = [version for version in successful_versions if version not in known_versions]
    failure_rate = len(non_draining_errors) / max(1, total_rows)
    cuda_growth = (cuda_after - cuda_before) if cuda_before is not None and cuda_after is not None else None
    resource_usage = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_growth_mb": rss_after - rss_before,
        "cuda_before_mb": cuda_before,
        "cuda_after_mb": cuda_after,
        "cuda_growth_mb": cuda_growth,
    }
    stage_gates = {
        "service_ready_before": health_before.get("ready") is True,
        "service_ready_after": health_after.get("ready") is True,
        "requests_completed": len(results) > 0,
        "reload_count": len(reload_events) >= config.min_reload_count,
        "reloads_ok": all(event["reload"].get("event", {}).get("status") == "ok" for event in reload_events),
        "resume_ok": all(event["resume"].get("management", {}).get("draining") is False for event in reload_events),
        "only_draining_errors": not non_draining_errors,
        "no_half_switch_versions": not invalid_versions,
        "failure_rate": failure_rate <= config.max_failure_rate,
        "rss_growth": resource_usage["rss_growth_mb"] <= config.max_rss_growth_mb,
        "cuda_growth": cuda_growth is None or cuda_growth <= config.max_cuda_growth_mb,
    }
    summary = {
        "stage": "stage54_long_running_service_stress",
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "health_before": health_before,
        "health_after": health_after,
        "request_count": len(results) + 1,
        "reload_count": len(reload_events),
        "snapshot_count": len(snapshots),
        "total_rows": total_rows,
        "error_types": error_types,
        "non_draining_errors": non_draining_errors,
        "successful_versions": successful_versions,
        "invalid_versions": invalid_versions,
        "resource_usage": resource_usage,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "running.json", {**summary, "status": "complete"})
    _json_dump(output / "resource_usage.json", resource_usage)
    _write_jsonl(output / "request_results.jsonl", [baseline, *results])
    _write_jsonl(output / "reload_events.jsonl", reload_events)
    _write_jsonl(output / "metrics_snapshots.jsonl", snapshots)
    with (output / "metrics_snapshots.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "rss_mb", "cuda_allocated_mb"])
        writer.writeheader()
        for row in snapshots:
            writer.writerow(
                {
                    "elapsed_seconds": row["elapsed_seconds"],
                    "rss_mb": row["rss_mb"],
                    "cuda_allocated_mb": row["cuda_allocated_mb"],
                }
            )
    return summary


def run_stage54_fake_long_running_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage54LongRunConfig = Stage54LongRunConfig(duration_seconds=1.0, request_interval_seconds=0.1, reload_interval_seconds=0.3, snapshot_interval_seconds=0.2),
) -> dict[str, Any]:
    service = build_stage51_fake_service(port=config.port)
    return run_stage54_long_running_stress(service=service, output_dir=output_dir, config=config)


def run_stage54_real_long_running_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage54LongRunConfig = Stage54LongRunConfig(),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> dict[str, Any]:
    service = build_stage52_real_managed_service(preferred_device=preferred_device, max_length=max_length, port=config.port)
    return run_stage54_long_running_stress(service=service, output_dir=output_dir, config=config)
