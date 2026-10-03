from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage45_adapter_package import (
    Stage45InferenceRequest,
)
from experiments.civilization_transformer_qwen3.backend.openai_compatible_runtime import (
    OpenAICompatibleRuntime,
    OpenAICompatibleRuntimeConfig,
)


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _request(control_mode: str = "full") -> Stage45InferenceRequest:
    return Stage45InferenceRequest(
        text="Which channel?",
        memory_items=("The channel is cobalt.",),
        rule_items=("Use verified memory.",),
        state_values=(1.0, 0.0, 0.5),
        answer_options=("cobalt", "amber"),
        control_mode=control_mode,
    )


def test_runtime_maps_provider_choice_without_claiming_adapter_execution(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")

    def opener(request, timeout):
        assert request.get_header("Authorization") == "Bearer secret-for-test"
        assert timeout == 10.0
        return _Response(
            json.dumps(
                {
                    "id": "response-1",
                    "model": "test-model",
                    "choices": [{"message": {"content": '{"option_id": 0}'}}],
                    "usage": {"total_tokens": 10},
                }
            ).encode()
        )

    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1",
            model="test-model",
            api_key_env="TEST_API_KEY",
            timeout_seconds=10.0,
        ),
        urlopen_fn=opener,
    )
    response = runtime.predict(_request())
    assert response.status == "ok" and response.predicted_option_id == 0
    assert response.trace["adapter_execution"] is False
    assert response.trace["hidden_states_available"] is False
    assert response.trace["ablation_controls_executed"] is False


def test_runtime_rejects_ablation_controls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1",
            model="test-model",
            api_key_env="TEST_API_KEY",
        )
    )
    with pytest.raises(ValueError, match="only control_mode='full'"):
        runtime.predict(_request("adapter_disabled"))


def _choice_response(content: str, *, finish_reason: str = "stop") -> _Response:
    return _Response(
        json.dumps(
            {
                "id": "response-1",
                "model": "test-model",
                "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
                "usage": {"total_tokens": 10},
            }
        ).encode()
    )


def test_runtime_retries_an_empty_provider_message(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    calls: list[dict] = []

    def opener(request, timeout):
        calls.append(json.loads(request.data.decode()))
        return _choice_response("" if len(calls) == 1 else '{"option_id": 1}', finish_reason="length")

    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1", model="test-model", api_key_env="TEST_API_KEY"
        ),
        urlopen_fn=opener,
    )
    response = runtime.predict(_request())
    assert response.predicted_option_id == 1
    assert response.trace["provider_attempts"] == 2
    assert response.trace["provider_empty_responses"] == 1
    assert response.trace["provider_finish_reason"] == "length"
    assert "Do not write reasoning" in calls[1]["messages"][1]["content"]
    assert "Do not write reasoning" not in calls[0]["messages"][1]["content"]


def test_runtime_requests_json_mode_and_falls_back_when_the_provider_rejects_it(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    calls: list[dict] = []

    def opener(request, timeout):
        payload = json.loads(request.data.decode())
        calls.append(payload)
        if "response_format" in payload:
            raise HTTPError(
                "https://example.test/v1/chat/completions",
                400,
                "Bad Request",
                None,
                io.BytesIO(b'{"error":"unsupported parameter: response_format"}'),
            )
        return _choice_response('{"option_id": 0}')

    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1", model="test-model", api_key_env="TEST_API_KEY"
        ),
        urlopen_fn=opener,
    )
    response = runtime.predict(_request())
    assert response.predicted_option_id == 0
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert response.trace["provider_json_object_mode"] is False


def test_runtime_uses_json_mode_by_default(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    seen: list[dict] = []

    def opener(request, timeout):
        seen.append(json.loads(request.data.decode()))
        return _choice_response('{"option_id": 1}')

    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1", model="test-model", api_key_env="TEST_API_KEY"
        ),
        urlopen_fn=opener,
    )
    response = runtime.predict(_request())
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert response.trace["provider_json_object_mode"] is True


def test_runtime_extra_body_reaches_the_provider(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    seen: list[dict] = []

    def opener(request, timeout):
        seen.append(json.loads(request.data.decode()))
        return _choice_response('{"option_id": 0}')

    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1",
            model="test-model",
            api_key_env="TEST_API_KEY",
            extra_body={"reasoning_effort": "none"},
        ),
        urlopen_fn=opener,
    )
    runtime.predict(_request())
    assert seen[0]["reasoning_effort"] == "none"


def test_runtime_reports_a_persistently_empty_provider(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-for-test")
    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1", model="test-model", api_key_env="TEST_API_KEY"
        ),
        urlopen_fn=lambda request, timeout: _choice_response("   "),
    )
    with pytest.raises(ValueError, match="provider response was empty"):
        runtime.predict(_request())


def test_runtime_requires_https_and_environment_key(monkeypatch):
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleRuntimeConfig(base_url="http://example.test/v1", model="m")
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    runtime = OpenAICompatibleRuntime(
        OpenAICompatibleRuntimeConfig(
            base_url="https://example.test/v1",
            model="m",
            api_key_env="MISSING_API_KEY",
        )
    )
    with pytest.raises(RuntimeError, match="MISSING_API_KEY"):
        runtime.predict(_request())
