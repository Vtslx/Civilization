from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage51_runtime_management import (
    Stage51ManagedInferenceService,
    Stage51RuntimeDescriptor,
    Stage51VersionedFakeRuntime,
    build_stage51_fake_service,
    run_stage51_management_smoke,
)
from experiments.civilization_transformer_qwen3.analysis.stage49_persistent_inference_service import Stage49ServiceConfig


def _payload() -> dict:
    return {
        "id": "stage51-test",
        "text": "Evaluate the operation.",
        "memory_items": ["memory evidence"],
        "rule_items": ["rule evidence"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def _post(url: str, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def _get(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode())


def test_stage51_reload_increments_runtime_version() -> None:
    service = build_stage51_fake_service()
    service.load_runtime()
    first = service.predict_record(_payload())[0]
    event = service.reload_runtime()
    second = service.predict_record(_payload())[0]
    assert event["status"] == "ok"
    assert second["runtime_version"] == first["runtime_version"] + 1
    assert service.management.reload_count == 1


def test_stage51_reload_failure_rolls_back_runtime() -> None:
    service = build_stage51_fake_service()
    service.load_runtime()
    before = service.management.runtime_version

    def failing_factory():
        raise RuntimeError("reload failed")

    event = service.reload_runtime(runtime_factory=failing_factory)
    assert event["status"] == "error"
    assert service.management.runtime_version == before
    assert service.management.reload_failures == 1
    row = service.predict_record(_payload())[0]
    assert row["status"] == "ok"


def test_stage51_drain_rejects_and_resume_accepts() -> None:
    service = build_stage51_fake_service()
    service.load_runtime()
    service.set_draining(True)
    assert service.predict_record(_payload())[0]["error_type"] == "ServiceDraining"
    service.set_draining(False)
    assert service.predict_record(_payload())[0]["status"] == "ok"


def test_stage51_http_admin_endpoints() -> None:
    service = build_stage51_fake_service(port=0)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = _get(f"{base}/health")
        before = _post(f"{base}/v1/predict", _payload())
        reload_response = _post(f"{base}/admin/reload")
        after = _post(f"{base}/v1/predict", _payload())
        drain = _post(f"{base}/admin/drain")
        draining = _post(f"{base}/v1/predict", _payload())
        resume = _post(f"{base}/admin/resume")
        status = _get(f"{base}/admin/status")
    finally:
        service.shutdown()
    assert health["stage"] == "stage51_runtime_management"
    assert reload_response["status"] == "ok"
    assert after["rows"][0]["runtime_version"] == before["rows"][0]["runtime_version"] + 1
    assert drain["management"]["draining"] is True
    assert draining["rows"][0]["error_type"] == "ServiceDraining"
    assert resume["management"]["draining"] is False
    assert status["management"]["reload_count"] == 1


def test_stage51_smoke_writes_summary(tmp_path) -> None:
    summary = run_stage51_management_smoke(output_dir=tmp_path, port=0)
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
