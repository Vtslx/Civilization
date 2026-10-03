from pathlib import Path

import torch

from experiments.civilization_transformer_qwen3.adapter import (
    CivilizationAdapter,
    CivilizationAdapterConfig,
    Qwen3MultiAdapterModel,
)
from experiments.civilization_transformer_qwen3.adapter.context_encoder import FrozenQwenContextEncoder
from experiments.civilization_transformer_qwen3.analysis.multilayer_adapter_benchmark import (
    _attention_sums_are_normalized_or_empty,
    run_qwen3_multilayer_adapter,
)
from experiments.civilization_transformer_qwen3.analysis.multilayer_adapter_training import (
    multilayer_checkpoint_contains_qwen_weights,
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


def test_attention_normalization_accepts_valid_and_empty_context_rows() -> None:
    attention = torch.tensor(
        [
            [[0.25, 0.75], [0.5, 0.5]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    assert _attention_sums_are_normalized_or_empty(attention)
    assert _attention_sums_are_normalized_or_empty(torch.zeros((2, 4, 0)))
    assert not _attention_sums_are_normalized_or_empty(torch.tensor([[[0.2, 0.2]]]))


def test_multilayer_zero_scale_matches_baseline_and_removes_hooks() -> None:
    backend = Qwen3Backend(MODEL_PATH, preferred_device="cpu")
    encoded, _ = backend.encode(["because pressure rises therefore output changes"], max_length=64)
    baseline = backend.inference_forward(encoded)
    adapters = {
        16: CivilizationAdapter(CivilizationAdapterConfig(target_layer=16, residual_scale_init=0.0)),
        24: CivilizationAdapter(CivilizationAdapterConfig(target_layer=24, residual_scale_init=0.0)),
    }
    model = Qwen3MultiAdapterModel(backend, adapters)
    datasets, _ = build_path_dependency_datasets(
        samples_per_label=1,
        max_seq_len=64,
        seed=202,
        scenarios=("memory_required_two_hop",),
    )
    context = FrozenQwenContextEncoder(backend).build_context(
        [datasets["memory_required_two_hop"][0]],
        encoded["attention_mask"],
    )

    output = model(encoded, context)

    assert set(output.traces) == {16, 24}
    assert torch.equal(output.logits, baseline.logits)
    assert all(torch.equal(left, right) for left, right in zip(output.hidden_states, baseline.hidden_states, strict=True))
    assert model.active_hook_count == 0


def test_multilayer_benchmark_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_multilayer_adapter(
        output_dir=tmp_path,
        model_path=MODEL_PATH,
        configs=("dual_16_24",),
        seeds=(202,),
        samples_per_label=2,
        train_per_label=1,
        stress_profiles=("direct_v1",),
        training_steps=1,
        preferred_device="cpu",
        evaluation_batch_size=10,
        evaluation_modes=("full", "adapter_disabled", "zero_scale"),
    )

    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "config_comparison.csv",
        "layer_probe_metrics.csv",
        "path_metrics.csv",
        "ablation_drop.csv",
        "cross_layer_consistency.csv",
        "trace_contribution.csv",
        "language_preservation.csv",
        "resource_usage.json",
        "failure_cases.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    assert summary["num_training_runs"] == 1
    assert summary["weights_unchanged"]
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["stage_gates"]["zero_and_disabled_equivalent"]
    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1
    assert not multilayer_checkpoint_contains_qwen_weights(checkpoints[0])
