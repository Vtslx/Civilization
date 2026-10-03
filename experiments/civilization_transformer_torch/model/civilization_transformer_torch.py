from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..state import StateConfig
from .ablation import CivilizationAblationConfig
from .civilization_block_torch import CivilizationBlockTorch, CivilizationTraceTorch
from .transformer_torch import TransformerConfigTorch


@dataclass
class CivilizationModelOutputTorch:
    logits: torch.Tensor
    hidden_states: list[torch.Tensor]
    attention_weights: list[torch.Tensor]
    final_hidden: torch.Tensor
    civilization_traces: list[CivilizationTraceTorch]
    injection_traces: list[dict] = field(default_factory=list)


class CivilizationTransformerTorch(nn.Module):
    def __init__(self, config: TransformerConfigTorch):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dim)
        self.blocks = nn.ModuleList([CivilizationBlockTorch(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_vectors: torch.Tensor | None = None,
        state: StateConfig | None = None,
        rule_vectors: torch.Tensor | None = None,
        ablation_config: CivilizationAblationConfig | None = None,
    ) -> CivilizationModelOutputTorch:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.max().item() >= self.config.vocab_size or input_ids.min().item() < 0:
            raise ValueError("input_ids contain token ids outside the vocabulary")
        hidden = self.token_embedding(input_ids)
        seq_len = hidden.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        device = hidden.device
        dtype = hidden.dtype
        memory = torch.zeros((0, self.config.model_dim), dtype=dtype, device=device) if memory_vectors is None else memory_vectors.to(device, dtype)
        rules = torch.zeros((0, self.config.model_dim), dtype=dtype, device=device) if rule_vectors is None else rule_vectors.to(device, dtype)
        active_state = state or StateConfig(rigor=0.5, creativity=0.5, defensiveness=0.5)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)
        hidden_states = [hidden]
        attentions: list[torch.Tensor] = []
        traces: list[CivilizationTraceTorch] = []
        for block in self.blocks:
            hidden, trace = block(hidden, memory, active_state, rules, ablation_config=ablation_config)
            hidden_states.append(hidden)
            attentions.append(trace.self_attention)
            traces.append(trace)

        logits = self.lm_head(hidden)
        return CivilizationModelOutputTorch(
            logits=logits,
            hidden_states=hidden_states,
            attention_weights=attentions,
            final_hidden=hidden,
            civilization_traces=traces,
        )
