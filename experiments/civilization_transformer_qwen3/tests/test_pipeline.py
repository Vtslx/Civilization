from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.pipeline import run_qwen3_hidden_state_baseline


import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    DEFAULT_MODEL_PATH,
    MISSING_MODEL_REASON,
    model_available,
)

MODEL_PATH = DEFAULT_MODEL_PATH
pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_qwen3_smoke_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_hidden_state_baseline(
        output_dir=tmp_path,
        model_path=MODEL_PATH,
        seeds=(202,),
        samples_per_label=2,
        train_per_label=1,
        max_lengths=(64,),
        batch_size=2,
        preferred_device="cpu",
        selected_layers=(0, 28),
        run_chat_sanity=False,
    )

    required = (
        "summary.json",
        "environment.json",
        "model_integrity.json",
        "layer_metrics.csv",
        "variant_metrics.csv",
        "seed_stability.csv",
        "classification_report.json",
        "confusion_by_layer.csv",
        "resource_usage.json",
        "failure_cases.json",
        "pca.png",
        "umap.png",
    )
    assert all((tmp_path / name).exists() for name in required)
    assert summary["num_metric_rows"] == 16
    assert summary["truncation_count"] == 0
    assert summary["stage_gates"]["model_integrity"]
    assert not summary["stage_gates"]["all_29_layers_exported"]
    assert summary["best_accuracy"] >= 0.0
