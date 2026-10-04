from __future__ import annotations

import torch

from civilization.engine.stages.binary_path_diagnostic_benchmark import (
    run_qwen3_binary_path_diagnostic,
)
from civilization.engine.stages.binary_path_diagnostic_data import (
    BINARY_BASE_DIAGNOSTIC_MODES,
    BINARY_DIAGNOSTIC_MODES,
    assert_binary_pairs_valid,
    build_combined_binary_path_diagnostic_pairs,
    build_binary_path_diagnostic_pairs,
    split_binary_pairs,
)
from civilization.engine.stages.binary_path_diagnostic_training import (
    _deterministic_binary_pair_sequence,
    _path_drop_loss,
)
from civilization.research.torch_line.analysis.dataset import LOGIC_LABELS
import pytest
from civilization.engine.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_binary_pairs_are_deterministic_and_context_isolated() -> None:
    first = build_binary_path_diagnostic_pairs(pairs_per_mode=12, seed=202)
    second = build_binary_path_diagnostic_pairs(pairs_per_mode=12, seed=202)
    assert first == second
    assert set(first) == set(BINARY_BASE_DIAGNOSTIC_MODES)
    combined = build_combined_binary_path_diagnostic_pairs(first, pairs_per_mode=12)
    combined_train, combined_test = split_binary_pairs(combined, train_pairs=8, held_out_pairs=4)
    assert not ({pair.pair_id for pair in combined_train} & {pair.pair_id for pair in combined_test})
    assert {pair.mode for pair in combined} == set(BINARY_BASE_DIAGNOSTIC_MODES)
    all_pairs = [pair for pairs in first.values() for pair in pairs]
    assert_binary_pairs_valid(all_pairs)
    forbidden = set(LOGIC_LABELS)
    for mode, pairs in first.items():
        assert mode in BINARY_DIAGNOSTIC_MODES
        train, test = split_binary_pairs(pairs, train_pairs=8, held_out_pairs=4)
        assert not ({pair.pair_id for pair in train} & {pair.pair_id for pair in test})
        assert not ({pair.surface_group_id for pair in train} & {pair.surface_group_id for pair in test})
        for pair in pairs:
            assert pair.full_sample.text == pair.counterfactual_sample.text
            assert len(pair.answer_options) == 2
            assert pair.expected_full_option_id != pair.expected_counterfactual_option_id
            text = " ".join(
                (
                    *pair.answer_options,
                    pair.full_sample.memory_target,
                    pair.full_sample.rule_target,
                    pair.counterfactual_sample.memory_target,
                    pair.counterfactual_sample.rule_target,
                )
            ).lower()
            assert not (forbidden & set(text.replace("_", " ").replace("-", " ").split()))
            if pair.pair_type == "memory_binary_pair":
                assert pair.full_sample.rule_target == pair.counterfactual_sample.rule_target
                assert pair.full_sample.memory_target != pair.counterfactual_sample.memory_target
            if pair.pair_type == "rule_binary_pair":
                assert pair.full_sample.memory_target == pair.counterfactual_sample.memory_target
                assert pair.full_sample.rule_target != pair.counterfactual_sample.rule_target


def test_binary_sequence_and_path_drop_loss_are_deterministic() -> None:
    pairs = build_binary_path_diagnostic_pairs(pairs_per_mode=12, seed=202)["memory_only_diagnostic"]
    assert _deterministic_binary_pair_sequence(pairs, 20, 202) == _deterministic_binary_pair_sequence(pairs, 20, 202)
    full_scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    weak_ablated = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
    strong_ablated = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    weak_loss = _path_drop_loss(full_scores, weak_ablated, targets, 0.2)
    strong_loss = _path_drop_loss(full_scores, strong_ablated, targets, 0.2)
    assert weak_loss > strong_loss


def test_binary_path_diagnostic_smoke_writes_required_artifacts(tmp_path) -> None:
    summary = run_qwen3_binary_path_diagnostic(
        output_dir=tmp_path / "out",
        seed=202,
        pairs_per_mode=12,
        train_pairs=8,
        held_out_pairs=4,
        stage_a_steps=1,
        stage_b_steps=1,
        gradient_accumulation=1,
        preferred_device="cpu",
        evaluation_batch_size=24,
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "binary_pair_metrics.csv",
        "pair_flip_metrics.csv",
        "binary_diagnostic_metrics.csv",
        "path_ablation_drop.csv",
        "answer_option_metrics.csv",
        "fixed_centroid_metrics.csv",
        "stage_a_vs_stage_b.csv",
        "wrong_context_metrics.csv",
        "trace_contribution.csv",
        "language_preservation.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
    }
    assert required.issubset({path.name for path in (tmp_path / "out").iterdir()})
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["stage_gates"]["separate_mode_checkpoints"]
    assert summary["weights_unchanged"]
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 8
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "qwen_state_dict" not in payload
