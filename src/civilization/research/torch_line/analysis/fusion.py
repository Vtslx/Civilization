from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..memory import MemoryItem
from ..rules import RuleResult
from ..state import StateConfig, StateGateTorch
from .dataset import LOGIC_LABELS
from .injection import LogicCodeInjector


@dataclass(frozen=True)
class FusionDecision:
    selected_target_label: str
    alpha: float
    strategy: str
    injection_layer: int
    blocked: bool
    block_reason: str | None
    trace: dict


class CivilizationFusionController:
    def __init__(
        self,
        centroids: dict[str, np.ndarray | torch.Tensor],
        model_dim: int,
        injection_layer: int,
        base_alpha: float = 0.25,
        max_alpha: float = 0.60,
    ):
        self.centroids = centroids
        self.model_dim = model_dim
        self.injection_layer = injection_layer
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.state_gate = StateGateTorch()

    def decide(
        self,
        baseline_label: str,
        memory_items: list[MemoryItem],
        state: StateConfig,
        rule_result: RuleResult,
        mode: str = "full_fusion",
        forced_target_label: str | None = None,
        force_alpha: float | None = None,
    ) -> FusionDecision:
        if mode not in {"codebook_only", "memory_guided", "state_guided", "rule_gated", "full_fusion", "alpha_zero", "blocked"}:
            raise ValueError("unknown fusion mode")
        memory_target, memory_scores = self._memory_target(memory_items, baseline_label)
        target = forced_target_label or baseline_label
        memory_contribution = "ignored"
        state_contribution = "ignored"
        rule_contribution = "ignored"

        if mode in {"memory_guided", "full_fusion"}:
            target = memory_target
            memory_contribution = f"selected {target} from memory scores"
        if mode == "codebook_only":
            target = forced_target_label or baseline_label

        alpha = self.base_alpha
        strategy = "residual_norm"
        if mode in {"state_guided", "full_fusion"}:
            alpha, strategy = self._state_alpha_strategy(state, alpha)
            state_contribution = f"alpha adjusted to {alpha:.4f} with {strategy}"
        if mode == "alpha_zero":
            alpha = 0.0
            strategy = "residual_norm"
            state_contribution = "alpha forced to zero"
        if force_alpha is not None:
            alpha = force_alpha

        blocked = False
        block_reason = None
        if mode in {"rule_gated", "full_fusion", "blocked"}:
            if rule_result.violations:
                blocked = True
                block_reason = "; ".join(rule_result.violations)
                rule_contribution = f"blocked by hard rule: {block_reason}"
            elif rule_result.conflicts:
                block_reason = "; ".join(rule_result.conflicts)
                rule_contribution = f"conflict recorded: {block_reason}"
                alpha *= 0.5
            elif rule_result.soft_matches:
                alpha *= 0.9
                rule_contribution = f"soft rule reduced alpha: {'; '.join(rule_result.soft_matches)}"
            else:
                rule_contribution = "rules passed"

        alpha = max(0.0, min(alpha, self.max_alpha))
        trace = {
            "mode": mode,
            "baseline_label": baseline_label,
            "selected_target_label": target,
            "alpha": alpha,
            "strategy": strategy,
            "injection_layer": self.injection_layer,
            "blocked": blocked,
            "block_reason": block_reason,
            "memory_contribution": memory_contribution,
            "memory_scores": memory_scores,
            "state_contribution": state_contribution,
            "state": {"rigor": state.rigor, "creativity": state.creativity, "defensiveness": state.defensiveness},
            "rule_contribution": rule_contribution,
            "rule": {
                "passed": rule_result.passed,
                "violations": rule_result.violations,
                "soft_matches": rule_result.soft_matches,
                "conflicts": rule_result.conflicts,
            },
        }
        return FusionDecision(target, alpha, strategy, self.injection_layer, blocked, block_reason, trace)

    def build_injector(self, decision: FusionDecision) -> LogicCodeInjector | None:
        if decision.blocked:
            return None
        return LogicCodeInjector(
            centroids=self.centroids,
            target_label=decision.selected_target_label,
            model_dim=self.model_dim,
            layer_index=decision.injection_layer,
            alpha=decision.alpha,
            strategy=decision.strategy,
            position="all",
        )

    def _memory_target(self, memory_items: list[MemoryItem], fallback: str) -> tuple[str, dict[str, float]]:
        scores = {label: 0.0 for label in LOGIC_LABELS}
        keywords = {
            "causality": ("cause", "causal", "therefore", "because", "leads"),
            "negation": ("not", "never", "deny", "reject", "negation"),
            "conflict": ("conflict", "contradict", "always", "never"),
            "priority": ("priority", "critical", "urgent", "override"),
            "condition": ("if", "then", "when", "condition", "threshold"),
        }
        for item in memory_items:
            text = f"{item.summary} {item.content} {item.relation_type}".lower()
            weight = max(item.priority, 0.0) * max(min(item.confidence, 1.0), 0.0)
            for label, tokens in keywords.items():
                scores[label] += weight * sum(1 for token in tokens if token in text)
        target = max(scores, key=scores.get)
        if scores[target] <= 0.0:
            target = fallback
        return target, scores

    def _state_alpha_strategy(self, state: StateConfig, alpha: float) -> tuple[float, str]:
        signals = self.state_gate.signals(state)
        adjusted = alpha * (0.75 + 0.35 * state.creativity - 0.25 * state.rigor + 0.15 * signals.memory_scale)
        strategy = "gated" if state.creativity > state.rigor else "residual_norm"
        if state.defensiveness > 0.75 or state.rigor > 0.75:
            adjusted = min(adjusted, alpha)
            strategy = "residual_norm"
        return max(0.05, min(adjusted, self.max_alpha)), strategy
