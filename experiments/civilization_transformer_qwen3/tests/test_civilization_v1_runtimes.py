"""Tests for the provider-agnostic runtime layer of the Civilization v1 SDK."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from civilization_v1 import (
    CivilizationRequest,
    EmbeddedCivilization,
    EmbeddedConfig,
    LocalTransformersRuntimeConfig,
    ProviderRuntimeConfig,
    capabilities_for,
    resolve_runtime,
    runtime_kinds,
)
from civilization_v1.runtimes import AdapterRuntimeConfig
from experiments.civilization_transformer_qwen3.analysis.stage45_adapter_package import (
    Stage45InferenceRequest,
)
from experiments.civilization_transformer_qwen3.analysis.stage60_queue_rate_limit import Stage60SlowFakeRuntime
from experiments.civilization_transformer_qwen3.backend.decision_prompt import parse_option_id
from experiments.civilization_transformer_qwen3.backend.transformers_chat_runtime import (
    TransformersChatRuntime,
    TransformersChatRuntimeConfig,
)


def _request() -> Stage45InferenceRequest:
    return Stage45InferenceRequest(
        text="Which channel is supported?",
        memory_items=("The channel is cobalt.",),
        rule_items=("Use verified evidence.",),
        state_values=(0.9, 0.1, 0.8),
        answer_options=("cobalt", "amber"),
    )


def test_registry_exposes_the_three_runtime_kinds():
    assert set(runtime_kinds()) == {"provider", "local_transformers", "local_adapter"}


def test_capabilities_state_what_each_runtime_can_claim():
    provider = capabilities_for("provider").to_dict()
    assert provider["hidden_states"] is False
    assert provider["adapter_execution"] is False
    assert provider["provider_agnostic"] is True
    assert provider["local_weights"] is False

    local = capabilities_for("local_transformers").to_dict()
    assert local["local_weights"] is True
    assert local["hidden_states"] is False
    assert local["adapter_execution"] is False

    adapter = capabilities_for("local_adapter").to_dict()
    assert adapter["hidden_states"] is True
    assert adapter["adapter_execution"] is True
    assert adapter["ablation_controls"] is False

    with pytest.raises(ValueError, match="unknown runtime kind"):
        capabilities_for("qwen_only")


def test_resolve_runtime_builds_provider_runtime_without_model_download():
    resolved = resolve_runtime(
        ProviderRuntimeConfig(base_url="https://provider.test/v1", model="any-model", api_key="k")
    )
    assert resolved.kind == "provider"
    assert resolved.capabilities.adapter_execution is False
    runtime = resolved.factory()
    assert type(runtime).__name__ == "OpenAICompatibleRuntime"
    assert runtime.config.model == "any-model"

    with pytest.raises(ValueError, match="unsupported runtime config"):
        resolve_runtime("https://provider.test/v1")


def test_local_transformers_config_never_claims_hidden_states():
    resolved = resolve_runtime(LocalTransformersRuntimeConfig(model_path="/models/anything"))
    assert resolved.kind == "local_transformers"
    assert resolved.capabilities.hidden_states is False
    assert resolved.capabilities.adapter_execution is False


def test_embedded_config_selects_runtime_and_validates_per_kind():
    assert EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.test/v1",
        provider_model="any-model",
    ).runtime_config() == ProviderRuntimeConfig(
        base_url="https://provider.test/v1", model="any-model"
    )

    loopback = EmbeddedConfig(
        runtime="provider",
        provider_base_url="http://127.0.0.1:8000/v1",
        provider_model="local-server",
    )
    assert loopback.runtime_config().base_url == "http://127.0.0.1:8000/v1"

    with pytest.raises(ValueError, match="HTTPS"):
        EmbeddedConfig(
            runtime="provider",
            provider_base_url="http://remote.test/v1",
            provider_model="any-model",
        )

    insecure = EmbeddedConfig(
        runtime="provider",
        provider_base_url="http://remote.test/v1",
        provider_model="any-model",
        allow_insecure_http=True,
    )
    assert insecure.allow_insecure_http is True

    local = EmbeddedConfig(runtime="local_transformers", local_model_path="/models/anything")
    assert local.runtime_config() == LocalTransformersRuntimeConfig(model_path="/models/anything")

    with pytest.raises(ValueError, match="local_model_path must be non-empty"):
        EmbeddedConfig(runtime="local_transformers")

    with pytest.raises(ValueError, match="package_manifest and centroid_bundle"):
        EmbeddedConfig(runtime="local_adapter", local_model_path="/models/anything")

    with pytest.raises(ValueError, match="unknown runtime"):
        EmbeddedConfig(runtime="qwen3")

    with pytest.raises(ValueError, match="loopback only"):
        EmbeddedConfig(
            runtime="provider",
            provider_base_url="https://provider.test/v1",
            provider_model="any-model",
            host="0.0.0.0",
        )


def test_adapter_runtime_config_is_the_only_hidden_state_kind():
    resolved = resolve_runtime(
        AdapterRuntimeConfig(
            package_manifest="manifest.json",
            centroid_bundle="centroid.pt",
            model_path="/models/anything",
        )
    )
    assert resolved.kind == "local_adapter"
    assert resolved.capabilities.hidden_states is True


def test_embedded_service_reports_capabilities_over_http(tmp_path: Path):
    with EmbeddedCivilization(
        EmbeddedConfig(
            runtime="provider",
            provider_base_url="https://provider.invalid/v1",
            provider_model="test-model",
            state_dir=str(tmp_path),
            bearer_token_env=None,
        ),
        runtime_factory=lambda: Stage60SlowFakeRuntime(0.0),
    ) as runtime:
        client = runtime.start()
        payload = client._request("GET", "/v1/capabilities")
        assert payload["status"] == "ok"
        assert payload["runtime_label"] == "civilization-v1:provider"
        capabilities = payload["capabilities"]
        assert capabilities["kind"] == "provider"
        assert capabilities["adapter_execution"] is False
        assert capabilities["hidden_states"] is False
        assert runtime.capabilities.kind == "provider"
        assert (tmp_path / "capabilities.json").is_file()
        prediction = runtime.predict(
            CivilizationRequest(
                text="Choose the supported operation.",
                answer_options=("approve", "reject"),
                session_id="runtime-layer",
            )
        )
        assert prediction.option_text in {"approve", "reject"}


def test_parse_option_id_accepts_documented_forms_and_refuses_ambiguity():
    options = ("cobalt", "amber")
    assert parse_option_id('{"option_id": 1}', options) == 1
    assert parse_option_id('here: "option_id": 0', options) == 0
    assert parse_option_id("1", options) == 1
    assert parse_option_id("amber", options) == 1
    with pytest.raises(ValueError, match="one valid option_id"):
        parse_option_id("cobalt and amber both look fine", options)
    with pytest.raises(ValueError, match="one valid option_id"):
        parse_option_id('{"option_id": 7}', options)
    with pytest.raises(ValueError, match="empty"):
        parse_option_id("   ", options)


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(message["content"] for message in messages) + "\nassistant:"

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=2048):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        return '{"option_id": 1}'


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def generate(self, **encoded):
        return torch.tensor([[1, 2, 3, 9, 9]])


def test_local_transformers_runtime_uses_shared_protocol_without_adapter_claims():
    runtime = TransformersChatRuntime(
        TransformersChatRuntimeConfig(model_path="/models/anything"),
        backend_factory=lambda: (_FakeTokenizer(), _FakeModel(), "cpu"),
    )
    response = runtime.predict(_request())
    assert response.status == "ok"
    assert response.predicted_option_id == 1
    assert response.scores["api_choice"] == [0.0, 1.0]
    assert response.trace["runtime"] == "transformers_local_chat"
    assert response.trace["adapter_execution"] is False
    assert response.trace["hidden_states_available"] is False
    with pytest.raises(ValueError, match="only control_mode='full'"):
        runtime.predict(
            Stage45InferenceRequest(
                text="Which channel is supported?",
                memory_items=(),
                rule_items=(),
                state_values=(0.5, 0.5, 0.5),
                answer_options=("cobalt", "amber"),
                control_mode="adapter_disabled",
            )
        )
