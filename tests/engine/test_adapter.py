from pathlib import Path

import torch

from civilization.engine.adapter import (
    CivilizationAdapter,
    CivilizationAdapterConfig,
    CivilizationAdapterContext,
    PathSpecificCivilizationAdapter,
    Qwen3AdapterModel,
)
from civilization.engine.backend import Qwen3Backend
from civilization.research.torch_line.model import CivilizationAblationConfig


import pytest
from civilization.engine.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def _context(batch: int = 2, memory_count: int = 2, rule_count: int = 3) -> CivilizationAdapterContext:
    return CivilizationAdapterContext(
        memory_vectors=torch.randn(batch, memory_count, 1024),
        memory_mask=torch.ones(batch, memory_count, dtype=torch.bool),
        state_values=torch.tensor([[0.9, 0.1, 0.8]] * batch),
        rule_vectors=torch.randn(batch, rule_count, 1024),
        rule_mask=torch.ones(batch, rule_count, dtype=torch.bool),
        attention_mask=torch.ones(batch, 5, dtype=torch.long),
    )


def test_adapter_shapes_attention_and_parameter_limit() -> None:
    adapter = CivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(2, 5, 1024)

    output, trace = adapter(hidden, _context())

    assert output.shape == hidden.shape
    assert trace.memory_attention.shape == (2, 5, 2)
    assert trace.rule_attention.shape == (2, 5, 3)
    assert torch.allclose(trace.memory_attention.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert torch.allclose(trace.rule_attention.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert adapter.trainable_parameter_count < 2_000_000
    assert torch.isfinite(output).all()


def test_adapter_zero_scale_and_disabled_are_exactly_equivalent() -> None:
    adapter = CivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.0))
    hidden = torch.randn(1, 4, 1024)
    context = _context(batch=1)

    zero_output, zero_trace = adapter(hidden, context)
    context.adapter_enabled = False
    disabled_output, disabled_trace = adapter(hidden, context)

    assert torch.equal(zero_output, hidden)
    assert torch.equal(disabled_output, hidden)
    assert zero_trace.delta_norm == 0.0
    assert disabled_trace.delta_norm == 0.0


def test_adapter_path_ablation_zeroes_corresponding_contribution() -> None:
    adapter = CivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(1, 4, 1024)

    no_memory = _context(batch=1)
    no_memory.ablation_config = CivilizationAblationConfig(use_memory_path=False)
    _, memory_trace = adapter(hidden, no_memory)
    no_state = _context(batch=1)
    no_state.ablation_config = CivilizationAblationConfig(use_state_path=False)
    _, state_trace = adapter(hidden, no_state)
    no_rule = _context(batch=1)
    no_rule.ablation_config = CivilizationAblationConfig(use_rule_path=False)
    _, rule_trace = adapter(hidden, no_rule)

    assert memory_trace.memory_contribution_norm == 0.0
    assert memory_trace.memory_attention.shape[-1] == 0
    assert state_trace.state_contribution_norm == 0.0
    assert rule_trace.rule_contribution_norm == 0.0
    assert rule_trace.rule_attention.shape[-1] == 0


def test_path_specific_adapter_shape_scales_and_path_ablation() -> None:
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(1, 4, 1024)

    output, trace = adapter(hidden, _context(batch=1))

    assert output.shape == hidden.shape
    assert trace.path_specific_adapter_version == "path_specific_v2"
    assert trace.memory_delta_norm > 0.0
    assert trace.rule_delta_norm > 0.0
    assert trace.base_delta_norm > 0.0
    assert abs(trace.memory_residual_scale - 0.1) < 1e-6
    assert abs(trace.rule_residual_scale - 0.1) < 1e-6

    no_memory = _context(batch=1)
    no_memory.ablation_config = CivilizationAblationConfig(use_memory_path=False)
    _, no_memory_trace = adapter(hidden, no_memory)
    assert no_memory_trace.memory_delta_norm == 0.0
    assert no_memory_trace.memory_contribution_norm == 0.0
    assert no_memory_trace.memory_attention.shape[-1] == 0

    no_rule = _context(batch=1)
    no_rule.ablation_config = CivilizationAblationConfig(use_rule_path=False)
    _, no_rule_trace = adapter(hidden, no_rule)
    assert no_rule_trace.rule_delta_norm == 0.0
    assert no_rule_trace.rule_contribution_norm == 0.0
    assert no_rule_trace.rule_attention.shape[-1] == 0


def test_path_specific_adapter_zero_scales_are_exactly_equivalent() -> None:
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.0))
    hidden = torch.randn(1, 4, 1024)

    output, trace = adapter(hidden, _context(batch=1))

    assert torch.equal(output, hidden)
    assert trace.delta_norm == 0.0
    assert trace.memory_delta_norm == 0.0
    assert trace.rule_delta_norm == 0.0


def test_path_specific_memory_and_rule_gradients_are_separable() -> None:
    hidden = torch.randn(1, 4, 1024)
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    memory_context = _context(batch=1, memory_count=2, rule_count=0)
    output, _trace = adapter(hidden, memory_context)
    output.float().square().mean().backward()
    memory_grad = adapter.memory_value.weight.grad.abs().sum()
    rule_grad = adapter.rule_value.weight.grad
    assert memory_grad > 0
    assert rule_grad is None or torch.equal(rule_grad, torch.zeros_like(rule_grad))

    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    rule_context = _context(batch=1, memory_count=0, rule_count=2)
    output, _trace = adapter(hidden, rule_context)
    output.float().square().mean().backward()
    rule_grad = adapter.rule_value.weight.grad.abs().sum()
    memory_grad = adapter.memory_value.weight.grad
    assert rule_grad > 0
    assert memory_grad is None or torch.equal(memory_grad, torch.zeros_like(memory_grad))


def test_qwen_hook_zero_scale_matches_baseline_and_does_not_leak() -> None:
    backend = Qwen3Backend(MODEL_PATH, preferred_device="cpu")
    encoded, _ = backend.encode(["because pressure rises therefore output changes"], max_length=64)
    baseline = backend.inference_forward(encoded)
    adapter = CivilizationAdapter(CivilizationAdapterConfig(target_layer=16, residual_scale_init=0.0))
    model = Qwen3AdapterModel(backend, adapter)
    context = _context(batch=1)
    context.attention_mask = encoded["attention_mask"]

    output = model(encoded, context)

    assert torch.equal(output.logits, baseline.logits)
    assert all(torch.equal(left, right) for left, right in zip(output.hidden_states, baseline.hidden_states, strict=True))
    assert model.active_hook_count == 0


def test_adapter_generation_removes_hook() -> None:
    backend = Qwen3Backend(MODEL_PATH, preferred_device="cpu")
    encoded, _ = backend.encode(["A causes B."], max_length=64)
    adapter = CivilizationAdapter(CivilizationAdapterConfig(target_layer=16, residual_scale_init=0.0))
    model = Qwen3AdapterModel(backend, adapter)
    context = _context(batch=1)
    context.attention_mask = encoded["attention_mask"]

    generated = model.generate(encoded, context, max_new_tokens=2)

    assert generated.shape[1] > encoded["input_ids"].shape[1]
    assert model.active_hook_count == 0
