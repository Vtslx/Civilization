from __future__ import annotations

from pathlib import Path

import torch

from experiments.civilization_transformer_qwen3.analysis.binary_path_diagnostic_data import (
    RULE_CONDITIONED_CONFLICT_MODE,
    assert_rule_conditioned_conflict_pairs_valid,
    build_rule_conditioned_conflict_pairs,
)
from experiments.civilization_transformer_qwen3.analysis.rule_conflict_curriculum import (
    run_qwen3_rule_conflict_curriculum,
)
import pytest
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_rule_conditioned_conflict_v2_keeps_rule_as_only_target_flip() -> None:
    pairs = build_rule_conditioned_conflict_pairs(pairs_per_mode=8, seed=202)
    assert_rule_conditioned_conflict_pairs_valid(pairs)
    pair = pairs[0]
    assert pair.mode == RULE_CONDITIONED_CONFLICT_MODE
    assert pair.full_sample.text == pair.counterfactual_sample.text
    assert pair.full_sample.memory_target == pair.counterfactual_sample.memory_target
    assert pair.full_sample.rule_target != pair.counterfactual_sample.rule_target
    assert pair.full_sample.state_target == pair.counterfactual_sample.state_target
    assert "route alpha" not in pair.full_sample.memory_target
    assert "route beta" not in pair.full_sample.memory_target


def test_rule_conflict_curriculum_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_rule_conflict_curriculum(
        output_dir=tmp_path / "out",
        seed=202,
        pairs_per_mode=12,
        train_pairs=8,
        held_out_pairs=4,
        memory_steps=1,
        rule_steps=1,
        conflict_steps=1,
        combined_steps=1,
        centroid_steps=1,
        preferred_device="cpu",
        evaluation_modes=("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "no_state_path", "wrong_context"),
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "curriculum_stage_metrics.csv",
        "projected_readout_metrics.csv",
        "raw_readout_metrics.csv",
        "fixed_centroid_metrics.csv",
        "path_ablation_drop.csv",
        "wrong_context_metrics.csv",
        "pair_flip_metrics.csv",
        "rehearsal_retention.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    checkpoint_paths = sorted((tmp_path / "out" / "checkpoints").glob("*.pt"))
    assert checkpoint_paths
    payload = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
