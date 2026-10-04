from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from civilization.engine.adapter import (
    CivilizationAdapterConfig,
    CivilizationAdapterContext,
    PathSpecificCivilizationAdapter,
)
from civilization.engine.stages.context_readout_alignment import PathReadoutProjector
from civilization.engine.stages.evidence_answer_training import _answer_scores
from civilization.engine.stages.hidden_states import last_non_padding_pool
from civilization.engine.stages.rule_conflict_group_recovery import (
    _build_group_splits,
    _counterfactual_group,
    _mapping_audit,
    _memory_delta_option_margin_loss,
    run_qwen3_rule_conflict_group_recovery,
)
from civilization.research.torch_line.model import CivilizationAblationConfig


def _context(batch: int = 5, memory_count: int = 1, rule_count: int = 1) -> CivilizationAdapterContext:
    return CivilizationAdapterContext(
        memory_vectors=torch.randn(batch, memory_count, 1024),
        memory_mask=torch.ones(batch, memory_count, dtype=torch.bool),
        state_values=torch.tensor([[0.85, 0.15, 0.85]] * batch),
        rule_vectors=torch.randn(batch, rule_count, 1024),
        rule_mask=torch.ones(batch, rule_count, dtype=torch.bool),
        attention_mask=torch.ones(batch, 4, dtype=torch.long),
    )


def test_rule_delta_is_differentiable_and_rule_ablation_zeroes_it() -> None:
    torch.manual_seed(202)
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(5, 4, 1024)
    output, trace = adapter(hidden, _context())
    assert output.shape == hidden.shape
    assert trace.rule_delta_tensor is not None
    assert trace.rule_delta_tensor.requires_grad
    assert trace.rule_delta_norm > 0.0

    no_rule = _context()
    no_rule.ablation_config = CivilizationAblationConfig(use_rule_path=False)
    _output, ablated = adapter(hidden, no_rule)
    assert ablated.rule_delta_tensor is not None
    assert torch.equal(ablated.rule_delta_tensor, torch.zeros_like(hidden))
    assert ablated.rule_delta_norm == 0.0


def test_rule_loss_reaches_rule_path_and_projector_not_memory() -> None:
    torch.manual_seed(202)
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    projector = PathReadoutProjector()
    hidden = torch.randn(5, 4, 1024)
    _output, trace = adapter(hidden, _context())
    pooled = last_non_padding_pool(trace.rule_delta_tensor, torch.ones(5, 4, dtype=torch.long))
    scores = _answer_scores(projector(pooled), torch.randn(5, 1024))
    loss = _memory_delta_option_margin_loss(scores, torch.arange(5), 0.25)
    loss.backward()
    assert adapter.rule_value.weight.grad is not None
    assert float(adapter.rule_value.weight.grad.abs().sum()) > 0.0
    assert adapter.rule_residual_scale.grad is not None
    assert float(adapter.rule_residual_scale.grad.abs().sum()) > 0.0
    assert adapter.memory_value.weight.grad is None or float(adapter.memory_value.weight.grad.abs().sum()) == 0.0
    assert projector.projection.weight.grad is not None


def test_rule_and_conflict_groups_have_strict_five_candidate_mapping() -> None:
    train, heldout, _manifest = _build_group_splits(
        local_samples_per_label=4,
        local_train_groups=2,
        seed=202,
        max_length=64,
    )
    rows, failures = _mapping_audit(train + heldout)
    assert not failures
    assert rows
    for group in train + heldout:
        assert len(group.pairs) == 5
        assert [pair.expected_full_option_id for pair in group.pairs] == [0, 1, 2, 3, 4]
        if group.group_type == "rule_necessity_group":
            assert len({pair.full_sample.memory_target for pair in group.pairs}) == 1
            assert len({pair.full_sample.state_target for pair in group.pairs}) == 1
        if group.group_type == "memory_rule_conflict_group":
            assert all(pair.full_sample.leakage_family.startswith("rule_conditioned_conflict_v2") for pair in group.pairs)
            assert len(
                {
                    row["state_hash"]
                    for row in rows
                    if row["surface_group_id"] == group.surface_group_id
                    and row["group_type"] == group.group_type
                }
            ) == 1


def test_counterfactual_group_swaps_samples_and_targets_without_invalid_pair() -> None:
    train, _heldout, _manifest = _build_group_splits(
        local_samples_per_label=4,
        local_train_groups=2,
        seed=202,
        max_length=64,
    )
    original = train[0]
    counterfactual = _counterfactual_group(original)
    assert len(counterfactual.pairs) == 5
    assert [pair.expected_full_option_id for pair in counterfactual.pairs] == [0, 1, 2, 3, 4]
    original_by_counterfactual_target = {
        pair.expected_counterfactual_option_id: pair for pair in original.pairs
    }
    for pair in counterfactual.pairs:
        source = original_by_counterfactual_target[pair.expected_full_option_id]
        assert pair.full_sample == source.counterfactual_sample
        assert pair.counterfactual_sample == source.full_sample


def test_stage41_real_qwen_smoke_writes_artifacts(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN3_STAGE41_SMOKE") != "1":
        pytest.skip("Set RUN_QWEN3_STAGE41_SMOKE=1 to run the real Qwen3 Stage 41 smoke.")
    checkpoint = os.environ.get("QWEN3_STAGE40_CHECKPOINT")
    if not checkpoint:
        pytest.skip("QWEN3_STAGE40_CHECKPOINT is required")
    summary = run_qwen3_rule_conflict_group_recovery(
        output_dir=tmp_path / "stage41",
        stage40_checkpoint=checkpoint,
        local_samples_per_label=4,
        local_train_groups=2,
        rule_steps=2,
        conflict_steps=2,
        combined_steps=2,
        projector_sanity_steps=2,
        preferred_device=os.environ.get("QWEN3_TEST_DEVICE", "cpu"),
        strict_stage_gates=False,
    )
    required = {
        "summary.json",
        "stage40_checkpoint_verification.json",
        "option_target_mapping.csv",
        "gradient_path_report.csv",
        "delta_tensor_audit.csv",
        "failure_cases.json",
    }
    assert required <= {path.name for path in (tmp_path / "stage41").iterdir()}
    assert summary["qwen_trainable_parameters"] == 0.0
    for checkpoint_file in (tmp_path / "stage41" / "checkpoints").rglob("*.pt"):
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        assert "qwen_state_dict" not in payload
        assert "model_state_dict" not in payload
