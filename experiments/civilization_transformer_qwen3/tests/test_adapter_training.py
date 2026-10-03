from pathlib import Path

import torch

from experiments.civilization_transformer_qwen3.adapter import CivilizationAdapterConfig
from experiments.civilization_transformer_qwen3.analysis.adapter_training import (
    checkpoint_contains_qwen_weights,
    run_qwen3_adapter_training,
    split_dependency_samples,
)
from experiments.civilization_transformer_qwen3.backend import Qwen3Backend
from experiments.civilization_transformer_torch.analysis.dataset import build_path_dependency_datasets


import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_adapter_training_keeps_qwen_frozen_and_writes_small_checkpoint(tmp_path: Path) -> None:
    backend = Qwen3Backend(MODEL_PATH, preferred_device="cpu")
    datasets, _ = build_path_dependency_datasets(
        samples_per_label=4,
        max_seq_len=32,
        seed=202,
        scenarios=("memory_required_two_hop", "state_required_disambiguation", "rule_required_priority"),
        stress_profile="direct_v1",
    )
    train_samples = []
    for samples in datasets.values():
        train, _ = split_dependency_samples(samples, train_groups=2, seed=202)
        train_samples.extend(train)

    model, _, result = run_qwen3_adapter_training(
        backend=backend,
        train_samples=train_samples,
        adapter_config=CivilizationAdapterConfig(target_layer=16),
        output_dir=tmp_path,
        seed=202,
        steps=4,
        gradient_accumulation=2,
    )

    assert result.adapter_parameter_count < 2_000_000
    assert result.qwen_trainable_parameter_count == 0
    assert result.qwen_gradients_present == 0
    assert not result.optimizer_contains_qwen_parameters
    assert result.fingerprint_unchanged
    assert backend.verify_weights_unchanged()
    assert not checkpoint_contains_qwen_weights(result.checkpoint_path)
    assert model.active_hook_count == 0
    assert model.adapter.residual_scale.item() != 0.0
    assert all(parameter.grad is None for parameter in backend.model.parameters())
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "adapter_state_dict",
        "diagnostic_heads_state_dict",
        "adapter_config",
        "training_metadata",
    }
