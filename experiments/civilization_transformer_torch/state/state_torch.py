from __future__ import annotations

from dataclasses import dataclass

import torch

from ..device import resolve_device


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


class StateEncoderTorch:
    def __init__(self, model_dim: int, seed: int = 23, device: str | torch.device | None = None):
        self.model_dim = model_dim
        self.device = resolve_device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.projection = torch.randn((3, model_dim), generator=generator, dtype=torch.float32) / (3**0.5)
        self.projection = self.projection.to(self.device)

    def encode(self, config: StateConfig) -> torch.Tensor:
        raw = torch.tensor([config.rigor, config.creativity, config.defensiveness], dtype=torch.float32, device=self.device)
        return raw @ self.projection


class StateGateTorch:
    def signals(self, config: StateConfig) -> StateSignals:
        memory_scale = 0.75 + 0.5 * config.rigor
        rule_scale = 0.75 + 0.75 * config.defensiveness
        temperature = max(0.2, 1.2 - 0.7 * config.rigor + 0.4 * config.creativity)
        attention_bias = 0.2 * config.rigor + 0.3 * config.defensiveness
        return StateSignals(memory_scale, rule_scale, temperature, attention_bias)

    def apply_to_memory(self, memory_vectors: torch.Tensor, config: StateConfig) -> torch.Tensor:
        return memory_vectors * self.signals(config).memory_scale

    def apply_to_rules(self, rule_vectors: torch.Tensor, config: StateConfig) -> torch.Tensor:
        return rule_vectors * self.signals(config).rule_scale
