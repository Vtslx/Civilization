from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from civilization.engine.stages.stage59_access_control_audit import (
    Stage59SecurityConfig,
    build_stage59_fake_service,
    redact_payload,
    run_stage59_security_smoke,
)


def _payload() -> dict:
    return {
        "id": "stage59-test",
        "text": "secret text should be redacted",
        "memory_items": ["secret memory"],
        "rule_items": ["secret rule"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full"],
    }


def _post(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, token: str | None = None) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode("utf-8"))


def test_stage59_redacts_text_and_context() -> None:
    redacted = redact_payload(_payload(), config=Stage59SecurityConfig())
    assert redacted["text"]["chars"] == len("secret text should be redacted")
    assert redacted["text"]["preview"] == ""
    assert "secret" not in json.dumps(redacted)
    assert redacted["memory_items"][0]["chars"] == len("secret memory")
    assert "sha256_16" in redacted["rule_items"][0]


def test_stage59_token_required_for_predict_and_admin(tmp_path) -> None:
    token = "token-123"
    audit = tmp_path / "audit.jsonl"
    service = build_stage59_fake_service(
        port=0,
        security=Stage59SecurityConfig(
            bearer_token=token,
            require_token_for_predict=True,
            require_token_for_admin=True,
            audit_log_path=str(audit),
        ),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = _get(f"{base}/health")
        try:
            _post(f"{base}/v1/predict", _payload())
            raise AssertionError("predict without token should fail")
        except HTTPError as error:
            assert error.code == 401
        predict = _post(f"{base}/v1/predict", _payload(), token=token)
        try:
            _post(f"{base}/admin/reload")
            raise AssertionError("admin without token should fail")
        except HTTPError as error:
            assert error.code == 401
        reload_response = _post(f"{base}/admin/reload", token=token)
        security_status = _get(f"{base}/admin/security", token=token)
    finally:
        service.shutdown()
    assert health["ready"]
    assert predict["status"] == "ok"
    assert reload_response["event"]["status"] == "ok"
    assert security_status["security"]["bearer_token_configured"]
    audit_text = audit.read_text(encoding="utf-8")
    assert "secret text should be redacted" not in audit_text
    assert "access_denied" in audit_text
    assert "access_allowed" in audit_text


def test_stage59_security_smoke_passes(tmp_path) -> None:
    summary = run_stage59_security_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["predict_requires_token"]
    assert summary["stage_gates"]["admin_requires_token"]
    assert summary["stage_gates"]["audit_redacts_text"]
    assert (tmp_path / "audit.jsonl").exists()
