from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from .stage52_real_runtime_reload_stress import _cuda_allocated_mb, _rss_mb, build_stage52_real_managed_service
from .stage51_runtime_management import build_stage51_fake_service


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage53_concurrent_reload_stress")


@dataclass(frozen=True)
class Stage53ConcurrentReloadConfig:
    pre_reload_requests: int = 2
    concurrent_requests: int = 6
    post_reload_requests: int = 2
    concurrency: int = 3
    controls: tuple[str, ...] = ("full", "no_memory_path", "adapter_disabled")
    request_timeout_seconds: float = 240.0
    max_rss_growth_mb: float = 768.0
    max_cuda_growth_mb: float = 768.0
    port: int = 0


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_payload(index: int, controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": f"stage53-{index:04d}",
        "text": "A controller must decide whether to approve the operation after reviewing the supplied evidence.",
        "memory_items": ["The operational evidence supports approving the action.", f"request marker {index}"],
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


def _predict(base_url: str, index: int, controls: tuple[str, ...], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = _post_json(f"{base_url}/v1/predict", _request_payload(index, controls), timeout=timeout)
        return {
            "request_index": index,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "ok",
            "payload": payload,
        }
    except Exception as error:
        return {
            "request_index": index,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _row_versions(result: dict[str, Any]) -> list[int]:
    payload = result.get("payload", {})
    versions: list[int] = []
    for row in payload.get("rows", []):
        version = row.get("runtime_version")
        if isinstance(version, int):
            versions.append(version)
    return versions


def _row_errors(result: dict[str, Any]) -> list[str]:
    payload = result.get("payload", {})
    return [str(row.get("error_type")) for row in payload.get("rows", []) if row.get("status") == "error"]


def run_stage53_concurrent_reload_stress(
    *,
    service,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage53ConcurrentReloadConfig = Stage53ConcurrentReloadConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    server = service.start_background()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        health_before = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
        for index in range(config.pre_reload_requests):
            results.append(_predict(base_url, index, config.controls, config.request_timeout_seconds))
        rss_before = _rss_mb()
        cuda_before = _cuda_allocated_mb()
        old_version = health_before.get("management", {}).get("runtime_version")
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = [
                executor.submit(
                    _predict,
                    base_url,
                    config.pre_reload_requests + index,
                    config.controls,
                    config.request_timeout_seconds,
                )
                for index in range(config.concurrent_requests)
            ]
            # Let some requests enter the service before draining.
            time.sleep(0.05)
            drain = _post_json(f"{base_url}/admin/drain", timeout=config.request_timeout_seconds)
            reload_response = _post_json(f"{base_url}/admin/reload", timeout=config.request_timeout_seconds)
            resume = _post_json(f"{base_url}/admin/resume", timeout=config.request_timeout_seconds)
            for future in as_completed(futures):
                results.append(future.result())
        for index in range(config.post_reload_requests):
            results.append(
                _predict(
                    base_url,
                    config.pre_reload_requests + config.concurrent_requests + index,
                    config.controls,
                    config.request_timeout_seconds,
                )
            )
        health_after = _get_json(f"{base_url}/health", timeout=config.request_timeout_seconds)
    finally:
        service.shutdown()
    elapsed = time.perf_counter() - started
    rss_after = _rss_mb()
    cuda_after = _cuda_allocated_mb()
    cuda_growth = (cuda_after - cuda_before) if cuda_before is not None and cuda_after is not None else None
    new_version = reload_response.get("event", {}).get("new_version")
    successful_versions = [version for result in results for version in _row_versions(result)]
    error_types = [error for result in results for error in _row_errors(result)]
    invalid_versions = [
        version
        for version in successful_versions
        if version not in {old_version, new_version}
    ]
    non_draining_errors = [error for error in error_types if error != "ServiceDraining"]
    stage_gates = {
        "service_ready_before": health_before.get("ready") is True,
        "service_ready_after": health_after.get("ready") is True,
        "reload_ok": reload_response.get("event", {}).get("status") == "ok",
        "resume_ok": resume.get("management", {}).get("draining") is False,
        "no_half_switch_versions": not invalid_versions,
        "only_draining_errors": not non_draining_errors,
        "post_reload_version_seen": new_version in successful_versions,
        "rss_growth": (rss_after - rss_before) <= config.max_rss_growth_mb,
        "cuda_growth": cuda_growth is None or cuda_growth <= config.max_cuda_growth_mb,
    }
    summary = {
        "stage": "stage53_concurrent_reload_stress",
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "health_before": health_before,
        "health_after": health_after,
        "drain": drain,
        "reload": reload_response,
        "resume": resume,
        "old_version": old_version,
        "new_version": new_version,
        "successful_versions": successful_versions,
        "error_types": error_types,
        "invalid_versions": invalid_versions,
        "resource_usage": {
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_growth_mb": rss_after - rss_before,
            "cuda_before_mb": cuda_before,
            "cuda_after_mb": cuda_after,
            "cuda_growth_mb": cuda_growth,
        },
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    _json_dump(output / "resource_usage.json", summary["resource_usage"])
    with (output / "request_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: int(item["request_index"])):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return summary


def run_stage53_fake_concurrent_reload_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage53ConcurrentReloadConfig = Stage53ConcurrentReloadConfig(),
) -> dict[str, Any]:
    service = build_stage51_fake_service(port=config.port)
    return run_stage53_concurrent_reload_stress(service=service, output_dir=output_dir, config=config)


def run_stage53_real_concurrent_reload_stress(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage53ConcurrentReloadConfig = Stage53ConcurrentReloadConfig(pre_reload_requests=1, concurrent_requests=3, post_reload_requests=1),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> dict[str, Any]:
    service = build_stage52_real_managed_service(preferred_device=preferred_device, max_length=max_length, port=config.port)
    return run_stage53_concurrent_reload_stress(service=service, output_dir=output_dir, config=config)
