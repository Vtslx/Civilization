from pathlib import Path

import torch

from experiments.civilization_transformer_torch.analysis.run_civilization_transformer_experiment import run_civilization_transformer_experiment
from experiments.civilization_transformer_torch.memory import MemoryEncoderTorch, MemoryItem
from experiments.civilization_transformer_torch.model import CivilizationTransformerTorch, TransformerConfigTorch
from experiments.civilization_transformer_torch.rules import RuleEngineTorch, RuleItem
from experiments.civilization_transformer_torch.state import StateConfig


def _config() -> TransformerConfigTorch:
    return TransformerConfigTorch(vocab_size=64, model_dim=12, hidden_dim=24, num_heads=3, num_layers=2, max_seq_len=16, seed=77)


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


def _memory(config: TransformerConfigTorch) -> torch.Tensor:
    return MemoryEncoderTorch(config.model_dim, device="cpu").encode(
        [
            MemoryItem("m1", "causal memory", "cause therefore effect", "causality", 1.0, 1.0),
            MemoryItem("m2", "priority memory", "critical priority override", "priority", 0.8, 1.0),
        ]
    )


def _rules(config: TransformerConfigTorch) -> torch.Tensor:
    return RuleEngineTorch(
        [
            RuleItem("r-hard", "hard", "unsafe", "block", 1.0, "test"),
            RuleItem("r-soft", "soft", "evidence", "prefer", 0.7, "test"),
        ],
        model_dim=config.model_dim,
        device="cpu",
    ).encode_vectors()


def test_civilization_transformer_exports_shapes_and_traces() -> None:
    config = _config()
    model = CivilizationTransformerTorch(config)
    output = model(_input_ids(), memory_vectors=_memory(config), state=StateConfig(0.8, 0.1, 0.9), rule_vectors=_rules(config))

    assert output.logits.shape == (1, 5, config.vocab_size)
    assert len(output.hidden_states) == config.num_layers + 1
    assert len(output.attention_weights) == config.num_layers
    assert len(output.civilization_traces) == config.num_layers
    for trace in output.civilization_traces:
        assert trace.memory_attention.shape == (1, 5, 2)
        torch.testing.assert_close(trace.memory_attention.sum(dim=-1), torch.ones((1, 5)), atol=1e-5, rtol=1e-5)
        assert trace.rule_influence_norm > 0.0


def test_civilization_transformer_handles_empty_memory_rules_and_state_changes() -> None:
    config = _config()
    model = CivilizationTransformerTorch(config)
    empty = torch.zeros((0, config.model_dim))
    strict = model(_input_ids(), memory_vectors=empty, state=StateConfig(1.0, 0.0, 1.0), rule_vectors=empty)
    creative = model(_input_ids(), memory_vectors=empty, state=StateConfig(0.0, 1.0, 0.0), rule_vectors=empty)
    with_rules = model(_input_ids(), memory_vectors=empty, state=StateConfig(1.0, 0.0, 1.0), rule_vectors=_rules(config))

    assert strict.logits.shape == creative.logits.shape
    assert torch.isfinite(strict.logits).all()
    assert torch.linalg.vector_norm(strict.logits - creative.logits).item() > 1e-6
    assert with_rules.civilization_traces[-1].rule_influence_norm > strict.civilization_traces[-1].rule_influence_norm


def test_civilization_transformer_experiment_single_seed_passes_gate(tmp_path: Path) -> None:
    summary = run_civilization_transformer_experiment(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        max_seq_len=18,
        device="cpu",
        run_regressions=False,
    )

    assert summary["trace_checks_passed"]
    assert summary["losses_decreased"]
    assert summary["stage11_regression"]["allows_hidden_state_injection_planning"]
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "runs.json").exists()
    assert (tmp_path / "variant_accuracy.csv").exists()
