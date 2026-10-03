from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from experiments.civilization_transformer_qwen3.analysis.peft_matched_baseline_benchmark import (
    CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
    LORA_ALPHA,
    LORA_RANK,
    PREFIX_LENGTH,
    LoRAUpdate,
    lora_parameter_count,
    prefix_parameter_count,
)


def _config():
    return SimpleNamespace(
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )


def test_lora_parameter_count_is_matched():
    count = lora_parameter_count(_config(), LORA_RANK)
    assert count == 3_727_360
    assert abs(count - CIVILIZATION_OPTIMIZED_PARAMETER_COUNT) / CIVILIZATION_OPTIMIZED_PARAMETER_COUNT < 0.01


def test_prefix_parameter_count_is_matched():
    count = prefix_parameter_count(_config(), PREFIX_LENGTH)
    assert count == 3_727_360
    assert abs(count - CIVILIZATION_OPTIMIZED_PARAMETER_COUNT) / CIVILIZATION_OPTIMIZED_PARAMETER_COUNT < 0.01


def test_lora_zero_initialization_preserves_base_output_and_receives_gradient():
    torch.manual_seed(1)
    base = nn.Linear(8, 12, bias=False)
    update = LoRAUpdate(8, 12, rank=2, alpha=4)
    inputs = torch.randn(3, 8)
    before = base(inputs)
    after = before + update(inputs)
    assert torch.equal(before, after)
    after.square().mean().backward()
    assert update.b.weight.grad is not None
    assert float(update.b.weight.grad.abs().sum()) > 0


def test_lora_scaling_matches_registered_contract():
    update = LoRAUpdate(8, 12, rank=LORA_RANK, alpha=LORA_ALPHA)
    assert update.scaling == 2.0
