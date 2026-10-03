from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transformer import TransformerBlock, TransformerConfig, softmax
from ..state import StateConfig, StateGate


@dataclass
class CivilizationTrace:
    self_attention: np.ndarray
    memory_attention: np.ndarray
    state_memory_scale: float
    state_rule_scale: float
    rule_influence_norm: float


class CivilizationBlock:
    def __init__(self, config: TransformerConfig, seed: int = 31):
        rng = np.random.default_rng(seed)
        self.block = TransformerBlock(config, rng)
        self.state_gate = StateGate()
        self.rule_projection = rng.normal(0.0, 1.0 / np.sqrt(config.model_dim), size=(config.model_dim, config.model_dim))

    def _memory_attention(self, hidden: np.ndarray, memory_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if memory_vectors.size == 0:
            empty = np.zeros((hidden.shape[0], hidden.shape[1], 0), dtype=float)
            return np.zeros_like(hidden), empty
        memory = np.broadcast_to(memory_vectors[None, :, :], (hidden.shape[0], *memory_vectors.shape))
        scores = hidden @ memory.transpose(0, 2, 1) / np.sqrt(hidden.shape[-1])
        attention = softmax(scores, axis=-1)
        return attention @ memory, attention

    def forward(
        self,
        hidden: np.ndarray,
        memory_vectors: np.ndarray,
        state: StateConfig,
        rule_vectors: np.ndarray,
    ) -> tuple[np.ndarray, CivilizationTrace]:
        state_signals = self.state_gate.signals(state)
        transformed, self_attention = self.block.forward(hidden)
        scaled_memory = self.state_gate.apply_to_memory(memory_vectors, state)
        memory_context, memory_attention = self._memory_attention(transformed, scaled_memory)
        if rule_vectors.size == 0:
            rule_context = np.zeros((hidden.shape[-1],), dtype=float)
        else:
            scaled_rules = self.state_gate.apply_to_rules(rule_vectors, state)
            rule_context = scaled_rules.mean(axis=0) @ self.rule_projection
        output = transformed + memory_context + rule_context[None, None, :]
        trace = CivilizationTrace(
            self_attention=self_attention,
            memory_attention=memory_attention,
            state_memory_scale=state_signals.memory_scale,
            state_rule_scale=state_signals.rule_scale,
            rule_influence_norm=float(np.linalg.norm(rule_context)),
        )
        return output, trace
