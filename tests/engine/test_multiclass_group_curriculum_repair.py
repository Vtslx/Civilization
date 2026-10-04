from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from civilization.engine.stages.evidence_answer_data import build_evidence_answer_samples
from civilization.engine.stages.multiclass_group_curriculum_repair import (
    GROUP_TYPES,
    build_surface_group_candidate_batches,
    run_qwen3_multiclass_group_curriculum_repair,
)
from civilization.engine.stages.multiclass_necessity_repair import (
    _stage37_path_specific_pairs,
    _stage37_path_specific_samples,
)
from civilization.engine.stages.necessity_alignment_data import build_necessity_pairs
from civilization.engine.stages.real_task_data import (
    build_local_semireal_task_records,
    records_to_logic_datasets,
)


def test_surface_group_candidate_batches_are_complete_and_isolated() -> None:
    records = build_local_semireal_task_records(3, seed=202, context_grounding_mode="grounded_v1")
    datasets, _tokenizer = records_to_logic_datasets(records, max_seq_len=64, context_grounding_mode="grounded_v1")
    samples = [
        sample
        for rows in datasets.values()
        for sample in _stage37_path_specific_samples(rows)
    ]
    pairs = _stage37_path_specific_pairs(build_necessity_pairs(build_evidence_answer_samples(samples, "local_semireal")))
    groups = build_surface_group_candidate_batches(pairs)
    assert groups
    assert {group.group_type for group in groups} == set(GROUP_TYPES)
    for group in groups[:24]:
        assert len(group.pairs) == 5
        assert len({pair.full_sample.text for pair in group.pairs}) == 1
        assert len({pair.expected_full_option_id for pair in group.pairs}) == 5
        assert len({pair.full_sample.expected_pattern for pair in group.pairs}) == 5
        if group.group_type == "memory_necessity_group":
            assert all(pair.full_sample.required_paths == ("memory",) for pair in group.pairs)
        if group.group_type == "rule_necessity_group":
            assert all(pair.full_sample.required_paths == ("rule",) for pair in group.pairs)
        if group.group_type == "memory_rule_conflict_group":
            assert all(pair.full_sample.required_paths == ("memory", "rule") for pair in group.pairs)


def test_multiclass_group_curriculum_smoke_writes_artifacts(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN3_STAGE38_SMOKE") != "1":
        pytest.skip("Stage 38 Qwen3 smoke runs real model forwards; set RUN_QWEN3_STAGE38_SMOKE=1 to execute it.")
    summary = run_qwen3_multiclass_group_curriculum_repair(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=4,
        local_train_groups=2,
        external_train_per_label=2,
        external_heldout_per_label=2,
        external_task_names=(),
        memory_steps=1,
        rule_steps=1,
        conflict_steps=1,
        combined_steps=1,
        full_hidden_steps=1,
        preferred_device="cpu",
        evaluation_batch_size=4,
        routes=("local_only_transfer",),
        evaluation_modes=("full",),
        group_evaluation_modes=("full", "no_memory_path", "no_rule_path"),
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "group_metrics.csv",
        "stage_gate_metrics.csv",
        "projected_readout_metrics.csv",
        "fixed_centroid_metrics.csv",
        "projected_full_hidden_metrics.csv",
        "necessity_pair_metrics.csv",
        "path_ablation_drop.csv",
        "wrong_context_metrics.csv",
        "route_comparison.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
        "truncation_cases.json",
        "dataset_manifest.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("multiclass_necessity_*_group_*_seed_202.pt"))
    assert checkpoints
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "full_hidden_projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
