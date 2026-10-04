from __future__ import annotations

from pathlib import Path

import torch

from civilization.engine.stages.full_hidden_centroid_alignment import (
    FullHiddenCentroidProjector,
    run_qwen3_full_hidden_centroid_alignment,
)
import pytest
from civilization.engine.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_full_hidden_centroid_projector_shape_and_parameters() -> None:
    projector = FullHiddenCentroidProjector()
    vector = torch.randn(2, 1024)
    output = projector(vector)
    assert output.shape == vector.shape
    assert sum(parameter.numel() for parameter in projector.parameters()) < 1_100_000


def test_full_hidden_centroid_alignment_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_full_hidden_centroid_alignment(
        output_dir=tmp_path / "out",
        seed=202,
        pairs_per_mode=6,
        train_pairs=4,
        held_out_pairs=2,
        alignment_steps=1,
        memory_steps=1,
        rule_steps=1,
        conflict_steps=1,
        combined_steps=1,
        centroid_steps=1,
        preferred_device="cpu",
        evaluation_modes=("full", "adapter_disabled", "zero_scale", "no_memory_path"),
        active_modes=("memory_only_diagnostic",),
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "full_hidden_centroid_metrics.csv",
        "projected_full_hidden_metrics.csv",
        "projected_readout_metrics.csv",
        "raw_readout_metrics.csv",
        "path_ablation_drop.csv",
        "pair_flip_metrics.csv",
        "centroid_stability.csv",
        "rehearsal_retention.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    checkpoint_paths = sorted((tmp_path / "out" / "checkpoints").glob("full_hidden_centroid_alignment_seed_*.pt"))
    assert checkpoint_paths
    payload = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "full_hidden_projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
