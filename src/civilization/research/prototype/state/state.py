from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateConfig:
    rigor: float
    creativity: float
    defensiveness: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass(frozen=True)
class StateSignals:
    memory_scale: float
    rule_scale: float
    temperature: float
    attention_bias: float


class StateEncoder:
    def __init__(self, model_dim: int, seed: int = 23):
        self.model_dim = model_dim
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(0.0, 1.0 / np.sqrt(3), size=(3, model_dim))

    def encode(self, config: StateConfig) -> np.ndarray:
        raw = np.array([config.rigor, config.creativity, config.defensiveness], dtype=float)
        return raw @ self.projection


class StateGate:
    def signals(self, config: StateConfig) -> StateSignals:
        memory_scale = 0.75 + 0.5 * config.rigor
        rule_scale = 0.75 + 0.75 * config.defensiveness
        temperature = max(0.2, 1.2 - 0.7 * config.rigor + 0.4 * config.creativity)
        attention_bias = 0.2 * config.rigor + 0.3 * config.defensiveness
        return StateSignals(memory_scale, rule_scale, temperature, attention_bias)

    def apply_to_memory(self, memory_vectors: np.ndarray, config: StateConfig) -> np.ndarray:
        return memory_vectors * self.signals(config).memory_scale

    def apply_to_rules(self, rule_vectors: np.ndarray, config: StateConfig) -> np.ndarray:
        return rule_vectors * self.signals(config).rule_scale
