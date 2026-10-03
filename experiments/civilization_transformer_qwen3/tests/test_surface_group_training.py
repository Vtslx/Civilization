from collections import Counter

import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.surface_group_training import (
    SurfaceGroupBatch,
    SurfaceGroupSampler,
    build_surface_group_batches,
    compute_surface_group_losses,
    surface_flip_checkpoint_contains_qwen_weights,
)
from experiments.civilization_transformer_qwen3.analysis.surface_flip_benchmark import (
    run_qwen3_surface_flip_repair,
)
from experiments.civilization_transformer_torch.analysis.dataset import (
    LOGIC_LABELS,
    build_path_dependency_datasets,
)
from experiments.civilization_transformer_qwen3.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def _groups(samples_per_label: int = 12):
    datasets, _ = build_path_dependency_datasets(
        samples_per_label=samples_per_label,
        max_seq_len=64,
        seed=202,
        stress_profile="direct_v1",
    )
    return build_surface_group_batches(
        [sample for samples in datasets.values() for sample in samples]
    )


def test_surface_group_batch_validates_complete_context_flip_group() -> None:
    groups = _groups(samples_per_label=2)
    assert groups
    assert all(len(group.samples) == 5 for group in groups)
    assert all({sample.label for sample in group.samples} == set(LOGIC_LABELS) for group in groups)
    assert all(len({sample.text for sample in group.samples}) == 1 for group in groups)
    with pytest.raises(ValueError, match="exactly five"):
        SurfaceGroupBatch(
            surface_group_id=groups[0].surface_group_id,
            scenario=groups[0].scenario,
            stress_profile=groups[0].stress_profile,
            samples=groups[0].samples[:-1],
        )


def test_surface_group_sampler_is_deterministic_and_weighted() -> None:
    groups = _groups()
    first = SurfaceGroupSampler(groups, seed=202).sequence(2000)
    second = SurfaceGroupSampler(groups, seed=202).sequence(2000)
    assert [group.surface_group_id for group in first] == [group.surface_group_id for group in second]
    counts = Counter(group.scenario for group in first)
    assert 0.36 <= counts["surface_invariant_label_flip"] / len(first) <= 0.44
    assert 0.21 <= counts["rule_required_priority"] / len(first) <= 0.29
    remaining = 1.0 - (
        counts["surface_invariant_label_flip"] + counts["rule_required_priority"]
    ) / len(first)
    assert 0.31 <= remaining <= 0.39


def test_surface_group_losses_reward_correct_alignment_and_penalize_collapse() -> None:
    prototypes = torch.eye(5, 8)
    labels = torch.arange(5)
    aligned = prototypes.clone()
    wrong = prototypes.roll(1, dims=0)
    aligned_losses = compute_surface_group_losses(
        [aligned, aligned],
        aligned,
        labels,
        prototypes,
    )
    wrong_losses = compute_surface_group_losses(
        [wrong, wrong],
        wrong,
        labels,
        prototypes,
    )
    collapsed = compute_surface_group_losses(
        [torch.ones_like(aligned), torch.ones_like(aligned)],
        torch.ones_like(aligned),
        labels,
        prototypes,
    )
    assert aligned_losses.group_target_alignment_loss < wrong_losses.group_target_alignment_loss
    assert aligned_losses.group_all_correct_loss < wrong_losses.group_all_correct_loss
    assert aligned_losses.context_delta_direction_loss < wrong_losses.context_delta_direction_loss
    assert collapsed.group_collapse_penalty > 0
    assert aligned_losses.group_context_separation_loss < collapsed.group_context_separation_loss
    assert torch.isfinite(aligned_losses.prototype_margin)


def test_surface_group_losses_backpropagate_finite_gradients() -> None:
    prototypes = torch.eye(5, 8)
    labels = torch.arange(5)
    updates = torch.randn(5, 8, requires_grad=True)
    deltas = torch.randn(5, 8, requires_grad=True)
    losses = compute_surface_group_losses([updates, updates], deltas, labels, prototypes)
    total = (
        losses.group_target_alignment_loss
        + losses.group_context_separation_loss
        + losses.group_all_correct_loss
        + losses.context_delta_direction_loss
        + losses.cross_layer_flip_consistency_loss
        + losses.group_collapse_penalty
    )
    total.backward()
    assert updates.grad is not None and torch.isfinite(updates.grad).all()
    assert deltas.grad is not None and torch.isfinite(deltas.grad).all()
    assert updates.grad.abs().sum() > 0
    assert deltas.grad.abs().sum() > 0


def test_surface_flip_repair_smoke_writes_artifacts(tmp_path) -> None:
    summary = run_qwen3_surface_flip_repair(
        output_dir=tmp_path,
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
        "surface_group_metrics.csv",
        "group_failure_distribution.csv",
        "scenario_flip_comparison.csv",
        "profile_flip_comparison.csv",
        "seed_stability.csv",
        "layer_probe_metrics.csv",
        "path_metrics.csv",
        "ablation_drop.csv",
        "cross_layer_flip_consistency.csv",
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
    assert not surface_flip_checkpoint_contains_qwen_weights(checkpoints[0])
