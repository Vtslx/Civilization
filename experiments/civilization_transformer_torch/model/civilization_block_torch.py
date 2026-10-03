from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .transformer_torch import TransformerBlockTorch, TransformerConfigTorch
from .ablation import CivilizationAblationConfig, default_ablation_config
from ..state import StateConfig, StateGateTorch


@dataclass
class CivilizationTraceTorch:
    self_attention: torch.Tensor
    memory_attention: torch.Tensor
    state_memory_scale: float
    state_rule_scale: float
    rule_influence_norm: float


class CivilizationBlockTorch(nn.Module):
    def __init__(self, config: TransformerConfigTorch):
        super().__init__()
        self.config = config
        self.block = TransformerBlockTorch(config)
        self.state_gate = StateGateTorch()
        self.rule_projection = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.state_projection = nn.Linear(3, config.model_dim, bias=False)

    def _memory_attention(self, hidden: torch.Tensor, memory_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if memory_vectors.numel() == 0:
            empty = torch.zeros((hidden.shape[0], hidden.shape[1], 0), dtype=hidden.dtype, device=hidden.device)
            return torch.zeros_like(hidden), empty
        memory = memory_vectors.to(hidden.device, hidden.dtype).unsqueeze(0).expand(hidden.shape[0], -1, -1)
        scores = torch.matmul(hidden, memory.transpose(-1, -2)) / (hidden.shape[-1] ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, memory), attention

    def forward(
        self,
        hidden: torch.Tensor,
        memory_vectors: torch.Tensor,
        state: StateConfig,
        rule_vectors: torch.Tensor,
        ablation_config: CivilizationAblationConfig | None = None,
    ) -> tuple[torch.Tensor, CivilizationTraceTorch]:
        ablation = default_ablation_config(ablation_config)
        state_signals = self.state_gate.signals(state)
        transformed, self_attention = self.block(hidden)
        if ablation.use_state_path:
            state_tensor = torch.tensor(
                [state.rigor, state.creativity, state.defensiveness],
                dtype=transformed.dtype,
                device=transformed.device,
            )
            state_context = self.state_projection(state_tensor)
        else:
            state_context = torch.zeros((transformed.shape[-1],), dtype=transformed.dtype, device=transformed.device)

        if ablation.use_memory_path:
            memory_input = memory_vectors.to(transformed.device, transformed.dtype)
            scaled_memory = self.state_gate.apply_to_memory(memory_input, state) if ablation.use_state_path else memory_input
            memory_context, memory_attention = self._memory_attention(transformed, scaled_memory)
        else:
            memory_context = torch.zeros_like(transformed)
            memory_attention = torch.zeros((transformed.shape[0], transformed.shape[1], 0), dtype=transformed.dtype, device=transformed.device)

        if rule_vectors.numel() == 0 or not ablation.use_rule_path:
            rule_context = torch.zeros((transformed.shape[-1],), dtype=transformed.dtype, device=transformed.device)
        else:
            scaled_rules = self.state_gate.apply_to_rules(rule_vectors.to(transformed.device, transformed.dtype), state)
            if not ablation.use_state_path:
                scaled_rules = rule_vectors.to(transformed.device, transformed.dtype)
            rule_context = self.rule_projection(scaled_rules.mean(dim=0))

        output = transformed + memory_context + rule_context.view(1, 1, -1) + state_context.view(1, 1, -1)
        trace = CivilizationTraceTorch(
            self_attention=self_attention,
            memory_attention=memory_attention,
            state_memory_scale=state_signals.memory_scale,
            state_rule_scale=state_signals.rule_scale,
            rule_influence_norm=float(torch.linalg.vector_norm(rule_context).detach().cpu().item()),
        )
        return output, trace
