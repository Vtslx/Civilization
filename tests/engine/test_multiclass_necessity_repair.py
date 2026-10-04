from __future__ import annotations

from pathlib import Path

import torch

from civilization.engine.stages.evidence_answer_data import build_evidence_answer_samples
from civilization.engine.stages.multiclass_necessity_repair import (
    run_qwen3_multiclass_necessity_repair,
)
from civilization.engine.stages.necessity_alignment_data import (
    assert_necessity_pairs_valid,
    build_necessity_pairs,
)
from civilization.engine.stages.real_task_data import (
    build_local_semireal_task_records,
    records_to_logic_datasets,
)
import pytest
from civilization.engine.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_multiclass_necessity_pairs_preserve_surface_and_flip_context() -> None:
    records = build_local_semireal_task_records(2, seed=202, context_grounding_mode="grounded_v1")
    datasets, _tokenizer = records_to_logic_datasets(records, max_seq_len=64, context_grounding_mode="grounded_v1")
    samples = [sample for rows in datasets.values() for sample in rows]
    pairs = build_necessity_pairs(build_evidence_answer_samples(samples, "local_semireal"))
    assert_necessity_pairs_valid(pairs)
    pair_types = {pair.pair_type for pair in pairs}
    assert {"memory_necessity_pair", "rule_necessity_pair", "memory_rule_conflict_pair"} <= pair_types
    for pair in pairs[:20]:
        assert pair.full_sample.text == pair.counterfactual_sample.text
        assert pair.expected_full_option_id != pair.expected_counterfactual_option_id


def test_multiclass_necessity_repair_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_multiclass_necessity_repair(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=4,
        local_train_groups=2,
        external_train_per_label=2,
        external_heldout_per_label=2,
        external_task_names=("glue_rte", "boolq"),
        memory_steps=2,
        rule_steps=2,
        conflict_steps=2,
        combined_steps=2,
        full_hidden_steps=2,
        gradient_accumulation=1,
        preferred_device="cpu",
        evaluation_batch_size=4,
        evaluation_modes=("full", "adapter_disabled", "zero_scale", "no_memory_path", "no_rule_path", "wrong_context"),
        enable_group_losses=False,
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "projected_readout_metrics.csv",
        "fixed_centroid_metrics.csv",
        "projected_full_hidden_metrics.csv",
        "path_ablation_drop.csv",
        "necessity_pair_metrics.csv",
        "wrong_context_metrics.csv",
        "route_comparison.csv",
        "local_retention.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
        "truncation_cases.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    checkpoint_paths = sorted((tmp_path / "out" / "checkpoints").glob("multiclass_necessity_*_full_hidden_alignment_seed_202.pt"))
    assert checkpoint_paths
    payload = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "full_hidden_projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
