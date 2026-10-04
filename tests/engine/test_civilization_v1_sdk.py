from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from civilization import (
    CivilizationAPIError,
    CivilizationClient,
    CivilizationRequest,
    EmbeddedCivilization,
    EmbeddedConfig,
    MemorySystem,
)
from civilization.engine.stages.stage60_queue_rate_limit import Stage60SlowFakeRuntime


def _request(request_id: str = "sdk-request", *, session_id: str = "sdk-session") -> CivilizationRequest:
    return CivilizationRequest(
        request_id=request_id,
        text="Choose the supported operation.",
        answer_options=("approve", "reject"),
        session_id=session_id,
        task_name="sdk_integration",
        memory_items=("The operation has verified evidence.",),
        rule_items=("Use grounded evidence.",),
        state_values=(0.9, 0.1, 0.8),
    )


def _embedded(tmp_path: Path) -> EmbeddedCivilization:
    return EmbeddedCivilization(
        EmbeddedConfig(
            provider_base_url="https://provider.invalid/v1",
            provider_model="test-model",
            state_dir=str(tmp_path),
            bearer_token_env="CIVILIZATION_TEST_TOKEN",
        ),
        runtime_factory=lambda: Stage60SlowFakeRuntime(0.0),
    )


def test_request_contract_is_production_full_only() -> None:
    source_options = ["approve", "reject"]
    request = CivilizationRequest(text="stable", answer_options=source_options)
    source_options[0] = "mutated"
    payload = _request().to_payload()

    assert request.answer_options == ("approve", "reject")
    assert payload["controls"] == ["full"]
    assert payload["orion_memory"] == {
        "session_id": "sdk-session",
        "read": True,
        "write": True,
        "include_global": False,
        "allow_resolved_global": False,
    }
    with pytest.raises(ValueError, match="exactly three"):
        CivilizationRequest(text="bad", answer_options=("a", "b"), state_values=(1.0, 2.0))


def test_client_security_defaults_and_secret_safe_repr() -> None:
    with pytest.raises(ValueError, match="plain HTTP"):
        CivilizationClient("http://service.example.com")

    client = CivilizationClient("https://service.example.com", token="do-not-display")
    assert "do-not-display" not in repr(client)
    assert "configured" in repr(client)


def test_sdk_http_and_embedded_production_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVILIZATION_TEST_TOKEN", "sdk-test-token")
    runtime = _embedded(tmp_path)
    client = runtime.start()
    try:
        assert client.ready()["ready"] is True

        unauthenticated = CivilizationClient(client.base_url)
        with pytest.raises(CivilizationAPIError) as denied:
            unauthenticated.predict(_request("denied"))
        assert denied.value.status_code == 401

        explicit = client.write_memory(
            session_id="sdk-session",
            memory_system=MemorySystem.EPISODIC,
            content="A prior operation completed successfully.",
            summary="prior successful operation",
            metadata={"origin": "sdk-test"},
        )
        assert explicit["memory_system"] == "episodic"

        prediction = client.predict(_request())
        assert prediction.request_id == "sdk-request"
        assert prediction.option_id in {0, 1}
        assert prediction.option_text in {"approve", "reject"}
        assert prediction.memory_trace["session_id"] == "sdk-session"
        assert prediction.raw["control_mode"] == "full"

        batch = client.batch([_request("sdk-batch-1"), _request("sdk-batch-2")])
        assert [item.request_id for item in batch] == ["sdk-batch-1", "sdk-batch-2"]
        assert all(item.raw["control_mode"] == "full" for item in batch)

        matches = client.read_memory(
            session_id="sdk-session",
            query="prior successful operation",
            memory_system=MemorySystem.EPISODIC,
        )
        assert any(item["cell"]["cell_id"] == explicit["cell_id"] for item in matches)

        submitted = client.submit_job(_request("sdk-job"))
        completed = client.wait_job(submitted.job_id, timeout=10.0, poll_interval=0.01)
        assert completed.status == "completed"
        result_page = client.job_result(completed.job_id, limit=10)
        assert result_page["status"] == "ok"

        exported = client.create_export([completed.job_id], export_id="sdk-export")
        assert exported["status"] == "created"
        packaged = client.create_package("sdk-export")
        expected_sha256 = packaged["package"]["sha256"]
        destination = tmp_path / "downloads" / "sdk-export.tar.gz"
        downloaded = client.download_package("sdk-export", destination, expected_sha256=expected_sha256)
        assert downloaded["verified"] is True
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256

        saved = client.save_global_store()
        loaded = client.load_global_store()
        assert saved["status"] == "ok"
        assert loaded["status"] == "ok"
    finally:
        runtime.shutdown()


def test_embedded_direct_prediction_without_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIVILIZATION_TEST_TOKEN", raising=False)
    runtime = _embedded(tmp_path)
    try:
        prediction = runtime.predict(_request("embedded-direct"))
        assert prediction.request_id == "embedded-direct"
        assert prediction.raw["control_mode"] == "full"
        client = runtime.start()
        assert client.memory_status()["session_count"] == 1
    finally:
        runtime.shutdown()
