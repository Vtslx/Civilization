from pathlib import Path

import torch

from civilization.engine.stages.context_readout_alignment import (
    PathReadoutProjector,
    run_qwen3_context_readout_alignment,
)
from civilization.engine.stages.binary_path_diagnostic_data import (
    build_binary_path_diagnostic_pairs,
)
from civilization.engine.stages.binary_path_diagnostic_training import (
    _make_adapter,
)
from civilization.engine.adapter import CivilizationAdapterConfig
import pytest
from civilization.engine.model_paths import (
    MISSING_MODEL_REASON,
    model_available,
)

pytestmark = pytest.mark.skipif(not model_available(), reason=MISSING_MODEL_REASON)


def test_binary_context_pairs_preserve_only_required_path() -> None:
    pairs = build_binary_path_diagnostic_pairs(pairs_per_mode=4, seed=202)
    memory = pairs["memory_only_diagnostic"][0]
    assert memory.full_sample.text == memory.counterfactual_sample.text
    assert memory.full_sample.memory_target != memory.counterfactual_sample.memory_target
    assert memory.full_sample.rule_target == memory.counterfactual_sample.rule_target

    rule = pairs["rule_only_diagnostic"][0]
    assert rule.full_sample.text == rule.counterfactual_sample.text
    assert rule.full_sample.rule_target != rule.counterfactual_sample.rule_target
    assert rule.full_sample.memory_target == rule.counterfactual_sample.memory_target


def test_path_readout_projector_shape_and_no_qwen_parameters() -> None:
    projector = PathReadoutProjector()
    vector = torch.randn(3, 1024)
    output = projector(vector)
    assert output.shape == vector.shape
    assert all(parameter.requires_grad for parameter in projector.parameters())
    assert sum(parameter.numel() for parameter in projector.parameters()) < 1_100_000


def test_adapter_variant_factory_supports_path_specific() -> None:
    adapter = _make_adapter(CivilizationAdapterConfig(), "path_specific_v2")
    assert adapter.__class__.__name__ == "PathSpecificCivilizationAdapter"


def test_context_readout_alignment_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_qwen3_context_readout_alignment(
        output_dir=tmp_path / "out",
        seed=202,
        modes=("memory_only_diagnostic", "rule_only_diagnostic"),
        pairs_per_mode=12,
        train_pairs=8,
        held_out_pairs=4,
        projector_steps=2,
        adapter_steps=2,
        centroid_steps=1,
        preferred_device="cpu",
    )
    required = {
        "summary.json",
        "training_runs.json",
        "loss_curves.csv",
        "readout_metrics.csv",
        "pair_metrics.csv",
        "resource_usage.json",
        "failure_cases.json",
        "dataset_manifest.json",
    }
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert summary["stage_gates"]["qwen_frozen"]
    assert summary["weights_unchanged"]
    checkpoint_names = {path.name for path in (tmp_path / "out" / "checkpoints").iterdir()}
    assert any("adapter" in name for name in checkpoint_names)
    assert any("projector" in name for name in checkpoint_names)
    checkpoint = torch.load(next((tmp_path / "out" / "checkpoints").glob("*projector*.pt")), map_location="cpu", weights_only=True)
    assert "projector_state_dict" in checkpoint
    assert "qwen_state_dict" not in checkpoint
