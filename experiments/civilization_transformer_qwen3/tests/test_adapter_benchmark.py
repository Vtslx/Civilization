import json
from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.adapter_benchmark import (
    run_qwen3_civilization_adapter,
)


import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_adapter_benchmark_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_civilization_adapter(
        output_dir=tmp_path,
        model_path=MODEL_PATH,
        layers=(16,),
        seeds=(202,),
        samples_per_label=2,
        train_per_label=1,
        stress_profiles=("direct_v1",),
        training_steps=1,
        preferred_device="cpu",
        evaluation_batch_size=5,
        evaluation_modes=("full", "adapter_disabled", "zero_scale"),
    )

    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "layer_comparison.csv",
        "path_metrics.csv",
        "ablation_drop.csv",
        "surface_group_flips.csv",
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
    assert json.loads((tmp_path / "summary.json").read_text())["main_layer"] == 16
