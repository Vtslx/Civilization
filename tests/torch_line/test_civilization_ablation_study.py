from pathlib import Path

import torch

from civilization.research.torch_line.model import CivilizationAblationConfig, CivilizationTransformerTorch, TransformerConfigTorch
from civilization.research.torch_line.memory import MemoryEncoderTorch, MemoryItem
from civilization.research.torch_line.rules import RuleEngineTorch, RuleItem
from civilization.research.torch_line.state import StateConfig
from civilization.research.torch_line.analysis.run_civilization_ablation_study import run_civilization_ablation_study


def test_ablation_config_defaults_enable_all_paths_and_losses() -> None:
    config = CivilizationAblationConfig()

    assert config.use_memory_path
    assert config.use_state_path
    assert config.use_rule_path
    assert config.use_chain_state_loss
    assert config.use_final_state_loss
    assert config.use_priority_control_loss
    assert config.use_centroid_separation_loss
    assert config.use_hard_negative_loss


def test_civilization_forward_ablation_paths_are_explicitly_disabled() -> None:
    model_config = TransformerConfigTorch(vocab_size=32, model_dim=16, hidden_dim=32, num_heads=4, num_layers=2, max_seq_len=12, seed=202)
    model = CivilizationTransformerTorch(model_config)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    memory = MemoryEncoderTorch(model_config.model_dim, device=torch.device("cpu")).encode([
        MemoryItem("m1", "logic memory", "cause bridge target", "causality", 1.0, 1.0)
    ])
    rules = RuleEngineTorch(
        [RuleItem("r1", "soft", "evidence", "prefer evidence", 1.0, "test")],
        model_dim=model_config.model_dim,
        device=torch.device("cpu"),
    ).encode_vectors()
    state = StateConfig(rigor=0.9, creativity=0.1, defensiveness=0.8)

    full = model(input_ids, memory_vectors=memory, state=state, rule_vectors=rules)
    no_memory = model(input_ids, memory_vectors=memory, state=state, rule_vectors=rules, ablation_config=CivilizationAblationConfig(use_memory_path=False))
    no_rule = model(input_ids, memory_vectors=memory, state=state, rule_vectors=rules, ablation_config=CivilizationAblationConfig(use_rule_path=False))
    no_state_strict = model(input_ids, memory_vectors=memory, state=state, rule_vectors=rules, ablation_config=CivilizationAblationConfig(use_state_path=False))
    no_state_creative = model(input_ids, memory_vectors=memory, state=StateConfig(rigor=0.1, creativity=0.9, defensiveness=0.1), rule_vectors=rules, ablation_config=CivilizationAblationConfig(use_state_path=False))

    assert full.civilization_traces[-1].memory_attention.shape[-1] == 1
    assert no_memory.civilization_traces[-1].memory_attention.shape[-1] == 0
    assert no_rule.civilization_traces[-1].rule_influence_norm == 0.0
    assert torch.linalg.vector_norm(no_state_strict.logits - no_state_creative.logits).item() == 0.0
    assert torch.isfinite(no_memory.logits).all()
    assert torch.isfinite(no_rule.logits).all()


def test_civilization_ablation_study_outputs_contribution_metrics(tmp_path: Path) -> None:
    summary = run_civilization_ablation_study(
        output_dir=tmp_path,
        modes=("full", "no_memory_path", "no_chain_state_loss", "no_priority_control_loss", "structure_only"),
        seeds=(202,),
        samples_per_label=40,
        train_per_label=25,
        seq_lens=(32,),
        model_sizes=("medium",),
        training_steps=30,
        device="cpu",
    )

    assert set(summary["modes"]) == {"full", "no_memory_path", "no_chain_state_loss", "no_priority_control_loss", "structure_only"}
    assert summary["mode_summaries"]["full"]["failures"]["nan_inf"] == 0
    assert summary["stage_gates"]["full_mode_passes"]
    assert "structure_only" in summary["comparison"]
    assert "two_hop_logic" in summary["comparison"]["no_chain_state_loss"]["scenario_drop"]
    assert "memory_path" in summary["weak_contribution_modules"]
    assert isinstance(summary["passes_stage_gate"], bool)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "ablation_metrics.csv").exists()
    assert (tmp_path / "mode_comparison.csv").exists()
    assert (tmp_path / "scenario_drop.csv").exists()
    assert (tmp_path / "trace_contribution.csv").exists()
    assert (tmp_path / "failure_cases.json").exists()
