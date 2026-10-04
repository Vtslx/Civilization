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
from civilization.engine.stages.memory_delta_gradient_diagnostic import (
    _assert_option_target_mapping,
    _memory_delta_option_margin_loss,
    _pooled_trace_delta,
    run_qwen3_memory_delta_gradient_diagnostic,
)
from civilization.engine.stages.memory_group_gate_repair import _build_memory_group_splits
from civilization.research.torch_line.model import CivilizationAblationConfig


def _context(batch: int = 5, memory_count: int = 2, rule_count: int = 0) -> CivilizationAdapterContext:
    return CivilizationAdapterContext(
        memory_vectors=torch.randn(batch, memory_count, 1024),
        memory_mask=torch.ones(batch, memory_count, dtype=torch.bool),
        state_values=torch.tensor([[0.9, 0.1, 0.8]] * batch),
        rule_vectors=torch.randn(batch, rule_count, 1024),
        rule_mask=torch.ones(batch, rule_count, dtype=torch.bool),
        attention_mask=torch.ones(batch, 4, dtype=torch.long),
    )


def test_path_specific_trace_exposes_differentiable_delta_tensors() -> None:
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(5, 4, 1024)
    output, trace = adapter(hidden, _context())

    assert output.shape == hidden.shape
    assert trace.memory_delta_tensor is not None
    assert trace.rule_delta_tensor is not None
    assert trace.state_delta_tensor is not None
    assert trace.base_delta_tensor is not None
    assert trace.memory_delta_tensor.shape == hidden.shape
    assert trace.memory_delta_tensor.requires_grad
    assert trace.memory_delta_norm > 0.0

    no_memory = _context()
    no_memory.ablation_config = CivilizationAblationConfig(use_memory_path=False)
    _output, no_memory_trace = adapter(hidden, no_memory)
    assert no_memory_trace.memory_delta_tensor is not None
    assert torch.equal(no_memory_trace.memory_delta_tensor, torch.zeros_like(hidden))
    assert no_memory_trace.memory_delta_norm == 0.0


def test_five_candidate_option_target_mapping_is_strict() -> None:
    train_groups, test_groups, _manifest = _build_memory_group_splits(
        local_samples_per_label=4,
        local_train_groups=2,
        seed=202,
        max_length=64,
    )
    ok, rows, failures = _assert_option_target_mapping(train_groups + test_groups)
    assert ok
    assert not failures
    assert rows
    for group_id in {row["surface_group_id"] for row in rows}:
        targets = [row["target_tensor"] for row in rows if row["surface_group_id"] == group_id]
        assert targets == [0, 1, 2, 3, 4]
        assert all(row["mapping_ok"] for row in rows if row["surface_group_id"] == group_id)


def test_memory_delta_margin_loss_reaches_memory_path_and_projector_only() -> None:
    torch.manual_seed(202)
    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    projector = PathReadoutProjector()
    hidden = torch.randn(5, 4, 1024)
    attention_mask = torch.ones(5, 4, dtype=torch.long)
    _output, trace = adapter(hidden, _context())
    assert trace.memory_delta_tensor is not None
    pooled_delta = last_non_padding_pool(trace.memory_delta_tensor, attention_mask)
    option_vectors = torch.randn(5, 1024)
    scores = _answer_scores(projector(pooled_delta), option_vectors)
    loss = _memory_delta_option_margin_loss(scores, torch.arange(5), margin=0.25)
    loss.backward()

    assert torch.isfinite(loss)
    assert adapter.memory_value.weight.grad is not None
    assert float(adapter.memory_value.weight.grad.abs().sum()) > 0.0
    assert adapter.memory_residual_scale.grad is not None
    assert float(adapter.memory_residual_scale.grad.abs().sum()) > 0.0
    assert projector.projection.weight.grad is not None
    assert float(projector.projection.weight.grad.abs().sum()) > 0.0
    assert adapter.rule_value.weight.grad is None or float(adapter.rule_value.weight.grad.abs().sum()) == 0.0


def test_pooled_trace_delta_reads_requested_path() -> None:
    class Output:
        pass

    adapter = PathSpecificCivilizationAdapter(CivilizationAdapterConfig(residual_scale_init=0.1))
    hidden = torch.randn(2, 3, 1024)
    _output, trace = adapter(hidden, _context(batch=2, memory_count=2, rule_count=0))
    fake = Output()
    fake.traces = {16: trace}
    fake.attention_mask = torch.ones(2, 3, dtype=torch.long)
    pooled = _pooled_trace_delta(fake, fake.attention_mask, "memory")
    assert pooled.shape == (2, 1024)
    assert torch.isfinite(pooled).all()


def test_memory_delta_gradient_diagnostic_smoke_writes_artifacts(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN3_STAGE40_SMOKE") != "1":
        pytest.skip("Stage 40 smoke loads Qwen3; set RUN_QWEN3_STAGE40_SMOKE=1 to execute it.")
    summary = run_qwen3_memory_delta_gradient_diagnostic(
        output_dir=tmp_path / "out",
        seed=202,
        local_samples_per_label=4,
        local_train_groups=2,
        repair_steps=1,
        preferred_device=os.environ.get("QWEN3_TEST_DEVICE", "cpu"),
    )
    required = {
        "summary.json",
        "option_target_mapping.csv",
        "candidate_score_audit.csv",
        "gradient_path_report.csv",
        "single_batch_loss_report.json",
        "delta_tensor_audit.csv",
        "projector_sanity_metrics.csv",
        "memory_path_only_metrics.csv",
        "memory_path_ablation_drop.csv",
        "memory_candidate_confusion.csv",
        "trace_contribution.csv",
        "resource_usage.json",
        "failure_cases.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["qwen_trainable_parameters"] == 0.0
    checkpoints = sorted((tmp_path / "out" / "checkpoints").glob("multiclass_necessity_*_memory_delta_path_only_seed_202.pt"))
    assert checkpoints
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert "adapter_state_dict" in payload
    assert "projector_state_dict" in payload
    assert "qwen_state_dict" not in payload
