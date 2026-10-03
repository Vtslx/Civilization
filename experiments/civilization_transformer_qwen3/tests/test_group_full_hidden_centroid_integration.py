from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from experiments.civilization_transformer_qwen3.analysis.full_hidden_centroid_alignment import FullHiddenCentroidProjector
from experiments.civilization_transformer_qwen3.analysis.group_full_hidden_centroid_integration import (
    _build_fixed_centroids,
    _load_stage41_checkpoint,
    _set_stage42_trainable,
    run_qwen3_group_full_hidden_centroid_integration,
)
from experiments.civilization_transformer_qwen3.analysis.rule_conflict_group_recovery import _build_group_splits


def test_full_hidden_projector_has_no_qwen_parameters() -> None:
    projector = FullHiddenCentroidProjector()
    assert sum(parameter.numel() for parameter in projector.parameters()) == 1024 * 1024 + 2 * 1024


def test_stage42_trainable_parameters_are_memory_rule_and_projectors_only() -> None:
    class DummyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory_value = torch.nn.Linear(2, 2)
            self.rule_value = torch.nn.Linear(2, 2)
            self.state_value = torch.nn.Linear(2, 2)
            self.base_value = torch.nn.Linear(2, 2)

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapters = torch.nn.ModuleDict({"16": DummyAdapter(), "24": DummyAdapter()})

    model = DummyModel()
    projector = torch.nn.Linear(2, 2)
    full_hidden_projector = torch.nn.Linear(2, 2)
    trainable = _set_stage42_trainable(model, projector, full_hidden_projector)
    trainable_ids = {id(parameter) for parameter in trainable}
    for name, parameter in model.named_parameters():
        if ".memory_" in name or ".rule_" in name:
            assert parameter.requires_grad
            assert id(parameter) in trainable_ids
        else:
            assert not parameter.requires_grad
    assert all(parameter.requires_grad for parameter in projector.parameters())
    assert all(parameter.requires_grad for parameter in full_hidden_projector.parameters())


def test_stage41_checkpoint_loader_rejects_qwen_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "qwen_state_dict": {},
            "adapter_state_dict": {},
            "projector_state_dict": {},
            "full_hidden_projector_state_dict": {},
            "training_metadata": {"adapter_variant": "path_specific_v2", "target_layers": [16, 24]},
        },
        checkpoint,
    )

    class Backend:
        device = torch.device("cpu")

    with pytest.raises(ValueError, match="forbidden Qwen"):
        _load_stage41_checkpoint(Backend(), checkpoint)  # type: ignore[arg-type]


def test_fixed_centroid_builder_uses_train_full_context_only(monkeypatch: pytest.MonkeyPatch) -> None:
    train, _heldout, _manifest = _build_group_splits(
        local_samples_per_label=4,
        local_train_groups=2,
        seed=202,
        max_length=64,
    )

    class Backend:
        device = torch.device("cpu")

    def fake_single_group_forward(**kwargs):
        group = kwargs["group"]
        targets = torch.tensor([pair.expected_full_option_id for pair in group.pairs], dtype=torch.long)
        pooled = torch.nn.functional.one_hot(targets, num_classes=5).float()
        return {"pooled": pooled, "targets": targets}

    monkeypatch.setattr(
        "experiments.civilization_transformer_qwen3.analysis.group_full_hidden_centroid_integration._single_group_forward",
        fake_single_group_forward,
    )
    monkeypatch.setattr(
        "experiments.civilization_transformer_qwen3.analysis.group_full_hidden_centroid_integration._option_vectors",
        lambda backend, group: torch.eye(5),
    )
    centroids, audit = _build_fixed_centroids(
        backend=Backend(),  # type: ignore[arg-type]
        model=object(),
        projector=object(),
        full_hidden_projector=object(),
        groups=train,
        max_length=64,
    )
    assert set(centroids) == {"memory_necessity_group", "rule_necessity_group", "memory_rule_conflict_group"}
    assert audit
    assert {row["source"] for row in audit} == {"train_full_context"}
    assert all(row["included_in_centroid"] is True for row in audit)


def test_stage42_real_qwen_smoke_writes_artifacts(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN3_STAGE42_SMOKE") != "1":
        pytest.skip("Set RUN_QWEN3_STAGE42_SMOKE=1 to run the real Qwen3 Stage 42 smoke.")
    checkpoint = os.environ.get("QWEN3_STAGE41_CHECKPOINT")
    if not checkpoint:
        pytest.skip("QWEN3_STAGE41_CHECKPOINT is required")
    summary = run_qwen3_group_full_hidden_centroid_integration(
        output_dir=tmp_path / "stage42",
        stage41_checkpoint=checkpoint,
        local_samples_per_label=4,
        local_train_groups=2,
        alignment_steps=2,
        preferred_device=os.environ.get("QWEN3_TEST_DEVICE", "cpu"),
        strict_stage_gates=False,
    )
    required = {
        "summary.json",
        "stage41_checkpoint_verification.json",
        "centroid_build_audit.csv",
        "fixed_centroid_metrics.csv",
        "projected_readout_retention.csv",
        "failure_cases.json",
    }
    assert required <= {path.name for path in (tmp_path / "stage42").iterdir()}
    assert summary["qwen_trainable_parameters"] == 0.0
    for checkpoint_file in (tmp_path / "stage42" / "checkpoints").rglob("*.pt"):
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        assert "qwen_state_dict" not in payload
        assert "model_state_dict" not in payload
