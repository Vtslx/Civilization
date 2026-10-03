from __future__ import annotations

import json
from urllib.request import Request, urlopen

from experiments.civilization_transformer_qwen3.analysis.stage49_persistent_inference_service import (
    Stage49FakeRuntime,
    Stage49PersistentInferenceService,
    Stage49ServiceConfig,
    run_stage49_service_smoke,
)


def _record(**overrides):
    payload = {
        "id": "request-1",
        "text": "Evaluate the operation.",
        "memory_items": ["memory evidence"],
        "rule_items": ["rule evidence"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
    }
    payload.update(overrides)
    return payload


def _post(url: str, payload: dict):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def _get(url: str):
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode())


def test_stage49_predict_record_expands_controls_and_audits() -> None:
    service = Stage49PersistentInferenceService(runtime_factory=Stage49FakeRuntime)
    rows = service.predict_record(_record(controls=["full", "no_memory_path", "zero_scale"]))
    assert [row["control_mode"] for row in rows] == ["full", "no_memory_path", "zero_scale"]
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["control_audit_passed"] for row in rows)
    assert service.metrics.prediction_rows == 3


def test_stage49_batch_isolates_bad_records() -> None:
    service = Stage49PersistentInferenceService(runtime_factory=Stage49FakeRuntime, config=Stage49ServiceConfig(max_batch_size=2))
    result = service.predict_batch([_record(), {"id": "bad", "text": "", "answer_options": []}])
    assert result["status"] == "ok"
    assert any(row["status"] == "ok" for row in result["rows"])
    assert any(row["status"] == "error" for row in result["rows"])


def test_stage49_batch_limit_fails_fast() -> None:
    service = Stage49PersistentInferenceService(runtime_factory=Stage49FakeRuntime, config=Stage49ServiceConfig(max_batch_size=1))
    try:
        service.predict_batch([_record(id="1"), _record(id="2")])
    except ValueError as error:
        assert "max_batch_size" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_stage49_http_health_predict_and_metrics() -> None:
    service = Stage49PersistentInferenceService(runtime_factory=Stage49FakeRuntime, config=Stage49ServiceConfig(port=0))
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = _get(f"{base}/health")
        predict = _post(f"{base}/v1/predict", _record(controls=["full", "adapter_disabled"]))
        metrics = _get(f"{base}/metrics")
    finally:
        service.shutdown()
    assert health["ready"] is True
    assert predict["status"] == "ok"
    assert len(predict["rows"]) == 2
    assert metrics["requests_received"] >= 1


def test_stage49_smoke_writes_summary(tmp_path) -> None:
    summary = run_stage49_service_smoke(output_dir=tmp_path, port=0)
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
