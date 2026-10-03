from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.civilization_transformer_torch.analysis import LogicCodeInjector
from experiments.civilization_transformer_torch.analysis.run_hidden_injection import run_hidden_injection_analysis
from experiments.civilization_transformer_torch.model import MiniTransformerTorch, TransformerConfigTorch


def _centroids(model_dim: int = 8) -> dict[str, np.ndarray]:
    labels = ("causality", "negation", "conflict", "priority", "condition")
    centroids = {}
    for index, label in enumerate(labels):
        vector = np.zeros(model_dim, dtype=np.float32)
        vector[index] = 1.0
        centroids[label] = vector
    return centroids


def test_injection_hook_none_keeps_default_outputs() -> None:
    config = TransformerConfigTorch(vocab_size=32, model_dim=8, hidden_dim=16, num_heads=2, num_layers=2, max_seq_len=8, seed=17)
    model = MiniTransformerTorch(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    baseline = model(input_ids)
    explicit_none = model(input_ids, hidden_injection_hook=None)

    torch.testing.assert_close(baseline.logits, explicit_none.logits)
    for left, right in zip(baseline.hidden_states, explicit_none.hidden_states):
        torch.testing.assert_close(left, right)
    assert explicit_none.injection_traces == []


def test_alpha_zero_injection_is_equivalent_and_traced() -> None:
    config = TransformerConfigTorch(vocab_size=32, model_dim=8, hidden_dim=16, num_heads=2, num_layers=2, max_seq_len=8, seed=17)
    model = MiniTransformerTorch(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    baseline = model(input_ids)
    injector = LogicCodeInjector(_centroids(config.model_dim), "causality", config.model_dim, layer_index=1, alpha=0.0)
    injected = model(input_ids, hidden_injection_hook=injector)

    torch.testing.assert_close(baseline.logits, injected.logits)
    for left, right in zip(baseline.hidden_states, injected.hidden_states):
        torch.testing.assert_close(left, right)
    assert len(injected.injection_traces) == 1
    assert injected.injection_traces[0]["alpha"] == 0.0
    assert injected.injection_traces[0]["delta_norm"] == 0.0


def test_injection_shape_validation_and_strong_warning() -> None:
    hidden = torch.ones((1, 3, 8))
    bad = _centroids(7)
    with pytest.raises(ValueError, match="injection vector shape"):
        LogicCodeInjector(bad, "causality", model_dim=8, layer_index=1)

    injector = LogicCodeInjector(_centroids(8), "causality", 8, layer_index=1, alpha=6.0, strategy="additive")
    injected, trace = injector(hidden, 1)
    assert injected.shape == hidden.shape
    assert torch.isfinite(injected).all()
    assert trace is not None
    assert trace["warning"] in {"alpha_too_high", "hidden_norm_ratio_exceeded"}
    assert trace["hidden_norm_ratio"] > 0.0


def test_hidden_injection_analysis_single_seed_passes_gate(tmp_path: Path) -> None:
    summary = run_hidden_injection_analysis(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        max_seq_len=18,
        device="cpu",
    )

    assert summary["stage11_regression"]["allows_hidden_state_injection_planning"]
    assert summary["zero_equivalence_rate"] == 1.0
    assert summary["target_hit_rate"] > 0.80
    assert summary["target_hit_not_worse_than_baseline"]
    assert summary["strong_warning_count"] > 0
    assert summary["correct_injection_norm_exceeded_count"] == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "injection_results.json").exists()
    assert (tmp_path / "injection_results.csv").exists()
