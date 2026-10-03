from pathlib import Path

import numpy as np

from experiments.civilization_transformer_torch.analysis import CivilizationFusionController
from experiments.civilization_transformer_torch.analysis.run_fusion_injection import run_fusion_injection_analysis
from experiments.civilization_transformer_torch.memory import MemoryItem
from experiments.civilization_transformer_torch.rules import RuleEngineTorch, RuleItem
from experiments.civilization_transformer_torch.state import StateConfig


def _centroids() -> dict[str, np.ndarray]:
    labels = ("causality", "negation", "conflict", "priority", "condition")
    result = {}
    for index, label in enumerate(labels):
        vector = np.zeros(8, dtype=np.float32)
        vector[index] = 1.0
        result[label] = vector
    return result


def _controller() -> CivilizationFusionController:
    return CivilizationFusionController(_centroids(), model_dim=8, injection_layer=1)


def _rules(text: str):
    engine = RuleEngineTorch(
        [
            RuleItem("hard", "hard", "unsafe", "block", 1.0, "test"),
            RuleItem("soft", "soft", "evidence", "prefer", 1.0, "test"),
            RuleItem("conflict", "conflict", "always|never", "record", 1.0, "test"),
        ],
        model_dim=8,
        device="cpu",
    )
    return engine.evaluate(text)


def test_fusion_controller_memory_state_and_rules_affect_decision() -> None:
    controller = _controller()
    memory = [MemoryItem("m1", "priority memory", "critical priority override", "priority", 1.0, 1.0)]
    strict = StateConfig(rigor=1.0, creativity=0.0, defensiveness=1.0)
    creative = StateConfig(rigor=0.0, creativity=1.0, defensiveness=0.0)

    memory_decision = controller.decide("causality", memory, strict, _rules("plain"), mode="memory_guided")
    strict_decision = controller.decide("causality", [], strict, _rules("plain"), mode="state_guided")
    creative_decision = controller.decide("causality", [], creative, _rules("plain"), mode="state_guided")
    soft_decision = controller.decide("causality", [], strict, _rules("evidence"), mode="rule_gated")
    conflict_decision = controller.decide("causality", [], strict, _rules("always and never"), mode="rule_gated")
    blocked_decision = controller.decide("causality", [], strict, _rules("unsafe"), mode="rule_gated")

    assert memory_decision.selected_target_label == "priority"
    assert strict_decision.alpha <= creative_decision.alpha
    assert creative_decision.strategy in {"gated", "residual_norm"}
    assert soft_decision.alpha < controller.base_alpha
    assert "conflict" in conflict_decision.trace["rule_contribution"]
    assert blocked_decision.blocked
    assert blocked_decision.block_reason
    assert controller.build_injector(blocked_decision) is None


def test_fusion_injection_analysis_single_seed_passes_gate(tmp_path: Path) -> None:
    summary = run_fusion_injection_analysis(
        output_dir=tmp_path,
        seeds=(202,),
        samples_per_label=30,
        train_per_label=20,
        max_seq_len=18,
        device="cpu",
        run_regressions=False,
    )

    assert summary["stage11_regression"]["allows_hidden_state_injection_planning"]
    assert summary["alpha_zero_equivalence_rate"] == 1.0
    assert summary["hard_rule_block_rate"] == 1.0
    assert summary["full_fusion_not_worse_than_codebook_only"]
    assert summary["wrong_label_drift_rate"] <= 0.50
    assert summary["hidden_norm_exceeded_count"] == 0
    assert summary["missing_trace_count"] == 0
    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "fusion_results.json").exists()
    assert (tmp_path / "fusion_results.csv").exists()
