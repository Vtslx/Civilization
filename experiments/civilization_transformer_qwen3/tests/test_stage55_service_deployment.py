from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.stage55_service_deployment import (
    Stage55DeploymentConfig,
    Stage55ServiceClient,
    run_stage55_fake_deployment_smoke,
    validate_stage55_deployment_bundle,
    write_stage55_deployment_bundle,
)
from experiments.civilization_transformer_qwen3.analysis.stage51_runtime_management import build_stage51_fake_service


def _payload() -> dict:
    return {
        "id": "stage55-test",
        "text": "Evaluate the operation.",
        "memory_items": ["memory evidence"],
        "rule_items": ["rule evidence"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "adapter_disabled"],
    }


def test_stage55_writes_nohup_deployment_bundle(tmp_path) -> None:
    output = tmp_path / "deploy"
    config = Stage55DeploymentConfig(
        output_dir=str(output),
        log_dir=str(output / "logs"),
        pid_file=str(output / "logs" / "service.pid"),
        log_file=str(output / "logs" / "service.log"),
        model_path="/models/qwen",
    )
    manifest = write_stage55_deployment_bundle(output_dir=output, config=config)
    validation = validate_stage55_deployment_bundle(output_dir=output)
    assert validation["passes_stage_gate"]
    start_script = Path(manifest["files"]["start_script"])
    assert os.access(start_script, os.X_OK)
    body = start_script.read_text(encoding="utf-8")
    assert "nohup" in body
    assert "HF_HUB_OFFLINE=1" in body
    assert "TRANSFORMERS_OFFLINE=1" in body
    assert "service.pid" in body
    assert (output / "example_request.json").exists()
    assert json.loads((output / "service_config.json").read_text(encoding="utf-8"))["model_path"] == "/models/qwen"


def test_stage55_client_calls_managed_fake_service() -> None:
    service = build_stage51_fake_service(port=0)
    server = service.start_background()
    host, port = server.server_address
    client = Stage55ServiceClient(f"http://{host}:{port}", timeout=10.0)
    try:
        health = client.health()
        prediction = client.predict(_payload())
        drain = client.drain()
        draining_prediction = client.predict(_payload())
        reload_response = client.reload()
        resume = client.resume()
        status = client.status()
    finally:
        service.shutdown()
    assert health["ready"]
    assert prediction["status"] == "ok"
    assert all(row["status"] == "ok" for row in prediction["rows"])
    assert drain["management"]["draining"] is True
    assert draining_prediction["rows"][0]["error_type"] == "ServiceDraining"
    assert reload_response["event"]["status"] == "ok"
    assert resume["management"]["draining"] is False
    assert status["management"]["runtime_version"] >= 2


def test_stage55_fake_deployment_smoke_passes(tmp_path) -> None:
    summary = run_stage55_fake_deployment_smoke(output_dir=tmp_path, port=0)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["deployment_bundle_valid"]
    assert summary["stage_gates"]["reload_ok"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "deployment_manifest.json").exists()
