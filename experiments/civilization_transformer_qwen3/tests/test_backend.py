from pathlib import Path

import pytest
import torch

from experiments.civilization_transformer_qwen3.backend import Qwen3Backend
from experiments.civilization_transformer_qwen3.backend.qwen3_backend import EXPECTED_SHA256


from experiments.civilization_transformer_qwen3.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


@pytest.fixture(scope="module")
def backend() -> Qwen3Backend:
    return Qwen3Backend(MODEL_PATH, preferred_device="cpu")


def test_qwen3_backend_loads_local_frozen_model_and_tokenizer(backend: Qwen3Backend) -> None:
    assert type(backend.model).__name__ == "Qwen3ForCausalLM"
    assert backend.config.hidden_size == 1024
    assert backend.config.num_hidden_layers == 28
    assert backend.config.vocab_size == 151936
    assert backend.config.num_attention_heads == 16
    assert backend.config.num_key_value_heads == 8
    assert backend.initial_sha256 == EXPECTED_SHA256
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())

    encoded = backend.tokenizer(["hello", "你好"], padding=True, return_tensors="pt")
    decoded = backend.tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)
    assert decoded[0]
    assert decoded[1]


def test_qwen3_backend_exports_logits_and_all_hidden_states(backend: Qwen3Backend) -> None:
    output, truncations = backend.forward_texts(["because pressure rises therefore output changes"], max_length=64)

    assert truncations == []
    assert output.logits.shape[:2] == output.input_ids.shape
    assert output.logits.shape[-1] == 151936
    assert len(output.hidden_states) == 29
    assert all(hidden.shape == (*output.input_ids.shape, 1024) for hidden in output.hidden_states)
    assert output.attention_mask.shape == output.input_ids.shape
    assert torch.isfinite(output.logits).all()
    assert all(torch.isfinite(hidden).all() for hidden in output.hidden_states)
    assert backend.verify_weights_unchanged()


def test_qwen3_backend_rejects_empty_text_and_records_truncation(backend: Qwen3Backend) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        backend.encode([""], max_length=64)

    _, truncations = backend.encode(["token " * 100], max_length=8)
    assert truncations
    assert truncations[0]["original_tokens"] > truncations[0]["truncated_tokens"]
    assert truncations[0]["core_logic_may_be_lost"]


def test_qwen3_parameter_fingerprint_is_stable(backend: Qwen3Backend) -> None:
    assert backend.parameter_fingerprint() == backend.parameter_fingerprint()
