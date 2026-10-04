from __future__ import annotations

from pathlib import Path

import torch

from civilization.engine.stages.projected_binary_path_necessity import (
    run_qwen3_projected_binary_path_necessity,
)
import pytest
from civilization.engine.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_projected_binary_path_necessity_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_projected_binary_path_necessity(
        output_dir=tmp_path / "out",
        seed=202,
        modes=("memory_only_diagnostic",),
        pairs_per_mode=4,
        train_pairs=2,
        held_out_pairs=2,
        stage_a_steps=1,
        stage_b_steps=1,
        stage_c_steps=1,
        preferred_device="cpu",
        evaluation_modes=("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "wrong_context"),
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "projected_readout_metrics.csv",
        "raw_readout_metrics.csv",
        "fixed_centroid_metrics.csv",
        "path_ablation_drop.csv",
        "wrong_context_metrics.csv",
        "pair_flip_metrics.csv",
        "stage_a_b_c_comparison.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    assert summary["readout"] == "projected_delta"

    checkpoint_paths = sorted((tmp_path / "out" / "checkpoints").glob("*/*.pt"))
    assert checkpoint_paths
    payload = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
