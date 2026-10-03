from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage59_access_control_audit import Stage59SecurityConfig
from experiments.civilization_transformer_qwen3.analysis.stage60_queue_rate_limit import (
    Stage60QueueConfig,
    build_stage60_fake_service,
    run_stage60_queue_smoke,
)


def _payload(index: int = 0) -> dict:
    return {
        "id": f"stage60-test-{index}",
        "text": "queue test",
        "memory_items": ["memory"],
        "rule_items": ["rule"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def _post(url: str, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def test_stage60_rate_limit_rejects_excess_requests(tmp_path) -> None:
    service = build_stage60_fake_service(
        port=0,
        security=Stage59SecurityConfig(audit_log_path=str(tmp_path / "audit.jsonl")),
        queue_config=Stage60QueueConfig(max_in_flight=2, per_client_qps=1, qps_window_seconds=60.0),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        first = _post(f"{base}/v1/predict", _payload(1))
        try:
            _post(f"{base}/v1/predict", _payload(2))
            raise AssertionError("second request should be rate limited")
        except HTTPError as error:
            assert error.code == 429
        queue = _get(f"{base}/admin/queue")
    finally:
        service.shutdown()
    assert first["status"] == "ok"
    assert queue["queue"]["metrics"]["rejected_rate_limit"] >= 1


def test_stage60_in_flight_overload_rejects_concurrent_requests(tmp_path) -> None:
    service = build_stage60_fake_service(
        port=0,
        security=Stage59SecurityConfig(audit_log_path=str(tmp_path / "audit.jsonl")),
        queue_config=Stage60QueueConfig(max_in_flight=1, acquire_timeout_seconds=0.01, per_client_qps=100),
        runtime_delay_seconds=0.2,
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    results: list[int] = []

    def worker(index: int) -> None:
        try:
            _post(f"{base}/v1/predict", _payload(index))
            results.append(200)
        except HTTPError as error:
            results.append(error.code)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(3)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        queue = _get(f"{base}/admin/queue")
    finally:
        service.shutdown()
    assert 200 in results
    assert any(code in {429, 503} for code in results)
    assert queue["queue"]["metrics"]["max_observed_in_flight"] <= 1


def test_stage60_queue_status_exposed_in_health(tmp_path) -> None:
    service = build_stage60_fake_service(
        port=0,
        security=Stage59SecurityConfig(audit_log_path=str(tmp_path / "audit.jsonl")),
        queue_config=Stage60QueueConfig(max_in_flight=1, per_client_qps=10),
    )
    server = service.start_background()
    host, port = server.server_address
    try:
        health = _get(f"http://{host}:{port}/health")
    finally:
        service.shutdown()
    assert health["stage"] == "stage60_queue_rate_limit"
    assert health["queue"]["config"]["max_in_flight"] == 1


def test_stage60_smoke_passes(tmp_path) -> None:
    summary = run_stage60_queue_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["rate_or_overload_rejected"]
    assert summary["stage_gates"]["max_in_flight_respected"]
