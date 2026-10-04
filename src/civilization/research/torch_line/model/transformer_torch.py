from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfigTorch:
    vocab_size: int = 64
    model_dim: int = 16
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")


@dataclass
class ModelOutputTorch:
    logits: torch.Tensor
    hidden_states: list[torch.Tensor]
    attention_weights: list[torch.Tensor]
    final_hidden: torch.Tensor
    injection_traces: list[dict] = field(default_factory=list)


HiddenInjectionHook = Callable[[torch.Tensor, int], tuple[torch.Tensor, dict | None] | torch.Tensor]


class TransformerBlockTorch(nn.Module):
    def __init__(self, config: TransformerConfigTorch):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.wk = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.wv = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.wo = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.model_dim),
        )
        self.norm2 = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, model_dim = hidden.shape
        head_count = self.config.num_heads
        head_dim = model_dim // head_count

        q = self.wq(hidden).view(batch, seq_len, head_count, head_dim).transpose(1, 2)
        k = self.wk(hidden).view(batch, seq_len, head_count, head_dim).transpose(1, 2)
        v = self.wv(hidden).view(batch, seq_len, head_count, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        if attention_bias is not None:
            scores = scores + attention_bias
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, seq_len, model_dim)

        hidden = self.norm1(hidden + self.wo(context))
        hidden = self.norm2(hidden + self.mlp(hidden))
        return hidden, attention


class MiniTransformerTorch(nn.Module):
    def __init__(self, config: TransformerConfigTorch):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dim)
        self.blocks = nn.ModuleList([TransformerBlockTorch(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        extra_tokens: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        hidden_injection_hook: HiddenInjectionHook | None = None,
    ) -> ModelOutputTorch:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.max().item() >= self.config.vocab_size or input_ids.min().item() < 0:
            raise ValueError("input_ids contain token ids outside the vocabulary")

        hidden = self.token_embedding(input_ids)
        if extra_tokens is not None:
            if extra_tokens.ndim == 2:
                extra_tokens = extra_tokens.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            if extra_tokens.shape[0] != hidden.shape[0] or extra_tokens.shape[2] != self.config.model_dim:
                raise ValueError("extra_tokens must have shape [batch, memory_seq, model_dim] or [memory_seq, model_dim]")
            hidden = torch.cat([hidden, extra_tokens.to(hidden.device, hidden.dtype)], dim=1)

        seq_len = hidden.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)
        hidden_states = [hidden]
        attentions: list[torch.Tensor] = []
        injection_traces: list[dict] = []
        for block_index, block in enumerate(self.blocks):
            hidden, attention = block(hidden, attention_bias=attention_bias)
            layer_index = block_index + 1
            if hidden_injection_hook is not None:
                injected = hidden_injection_hook(hidden, layer_index)
                if isinstance(injected, tuple):
                    hidden, trace = injected
                    if trace is not None:
                        injection_traces.append(trace)
                else:
                    hidden = injected
            hidden_states.append(hidden)
            attentions.append(attention)

        logits = self.lm_head(hidden)
        return ModelOutputTorch(logits=logits, hidden_states=hidden_states, attention_weights=attentions, final_hidden=hidden, injection_traces=injection_traces)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
