from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.evidence_answer_data import build_evidence_answer_samples
from experiments.civilization_transformer_qwen3.analysis.memory_group_gate_repair import (
    _build_memory_group_splits,
    _memory_group_losses,
    run_qwen3_memory_group_gate_repair,
)
from experiments.civilization_transformer_qwen3.analysis.multiclass_group_curriculum_repair import (
    build_surface_group_candidate_batches,
)
from experiments.civilization_transformer_qwen3.analysis.multiclass_necessity_repair import (
    _stage37_path_specific_pairs,
    _stage37_path_specific_samples,
)
from experiments.civilization_transformer_qwen3.analysis.necessity_alignment_data import build_necessity_pairs
from experiments.civilization_transformer_qwen3.analysis.real_task_data import (
    build_local_semireal_task_records,
    records_to_logic_datasets,
)


def test_memory_groups_are_complete_and_train_test_isolated() -> None:
    train_groups, test_groups, _manifest = _build_memory_group_splits(
        local_samples_per_label=4,
        local_train_groups=2,
        seed=202,
        max_length=64,
    )
    assert train_groups
    assert test_groups
    assert not ({group.surface_group_id for group in train_groups} & {group.surface_group_id for group in test_groups})
    for group in train_groups + test_groups:
        assert group.group_type == "memory_necessity_group"
        assert len(group.pairs) == 5
        assert len({pair.full_sample.text for pair in group.pairs}) == 1
        assert len({pair.expected_full_option_id for pair in group.pairs}) == 5
        assert all(pair.full_sample.required_paths == ("memory",) for pair in group.pairs)
        assert all(pair.full_sample.rule_target == pair.counterfactual_sample.rule_target for pair in group.pairs)
        assert all(pair.full_sample.memory_target != pair.counterfactual_sample.memory_target for pair in group.pairs)


def test_stage39_rejects_incomplete_memory_group() -> None:
    records = build_local_semireal_task_records(2, seed=202, context_grounding_mode="grounded_v1")
    datasets, _tokenizer = records_to_logic_datasets(records, max_seq_len=64, context_grounding_mode="grounded_v1")
    samples = [sample for rows in datasets.values() for sample in _stage37_path_specific_samples(rows)]
    pairs = _stage37_path_specific_pairs(build_necessity_pairs(build_evidence_answer_samples(samples, "local_semireal")))
    groups = [group for group in build_surface_group_candidate_batches(pairs) if group.group_type == "memory_necessity_group"]
    assert groups
    broken_pairs = list(groups[0].pairs[:-1])
    with pytest.raises(ValueError, match="exactly five"):
        type(groups[0])(group_type=groups[0].group_type, surface_group_id=groups[0].surface_group_id, pairs=tuple(broken_pairs))


def test_memory_group_loss_has_finite_nonzero_gradient() -> None:
    torch.manual_seed(202)
    scores = torch.randn(5, 5, requires_grad=True)
    no_memory_scores = scores.detach().clone() - 0.3
    no_rule_scores = scores.detach().clone()
    wrong_scores = scores.detach().clone() - 0.1
    projected = torch.randn(5, 1024, requires_grad=True)
    memory_vectors = torch.randn(5, 1024)
    targets = torch.arange(5)
    zero = scores.sum() * 0.0
    base = {
        "scores": scores,
        "targets": targets,
        "projected": projected,
        "fixed_loss": zero,
        "dynamic_loss": zero,
    }
    losses = _memory_group_losses(
        full=base,
        no_memory={**base, "scores": no_memory_scores},
        no_rule={**base, "scores": no_rule_scores},
        wrong={**base, "scores": wrong_scores},
        memory_vectors=memory_vectors,
        full_hidden_weight=0.25,
    )
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert scores.grad is not None
    assert projected.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(projected.grad).all()
    assert float(scores.grad.abs().sum()) > 0.0
    assert float(projected.grad.abs().sum()) > 0.0


def test_memory_group_gate_repair_smoke_writes_artifacts(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN3_STAGE39_SMOKE") != "1":
        pytest.skip("Stage 39 smoke loads Qwen3; set RUN_QWEN3_STAGE39_SMOKE=1 to execute it.")
    summary = run_qwen3_memory_group_gate_repair(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=4,
        local_train_groups=2,
        memory_stage_steps=1,
        full_hidden_alignment_steps=1,
        preferred_device=os.environ.get("QWEN3_TEST_DEVICE", "cpu"),
        strict_stage_gates=False,
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "memory_group_metrics.csv",
        "memory_context_diagnostics.csv",
        "memory_path_ablation_drop.csv",
        "memory_candidate_confusion.csv",
        "fixed_centroid_metrics.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["weights_unchanged"]
    assert summary["qwen_trainable_parameters"] == 0.0
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("multiclass_necessity_*_group_*_seed_202.pt"))
    assert checkpoints
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "full_hidden_projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
