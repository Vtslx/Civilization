from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from civilization.research.torch_line.model import CivilizationAblationConfig


@dataclass(frozen=True)
class CivilizationAdapterConfig:
    hidden_size: int = 1024
    bottleneck_size: int = 128
    target_layer: int = 16
    dropout: float = 0.0
    residual_scale_init: float = 0.0
    use_memory_path: bool = True
    use_state_path: bool = True
    use_rule_path: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CivilizationAdapterContext:
    memory_vectors: torch.Tensor
    memory_mask: torch.Tensor
    state_values: torch.Tensor
    rule_vectors: torch.Tensor
    rule_mask: torch.Tensor
    attention_mask: torch.Tensor
    ablation_config: CivilizationAblationConfig | None = None
    adapter_enabled: bool = True
    force_zero_scale: bool = False


@dataclass
class CivilizationAdapterTrace:
    target_layer: int
    pre_hidden_norm: float
    post_hidden_norm: float
    delta_norm: float
    hidden_norm_ratio: float
    memory_attention: torch.Tensor
    memory_contribution_norm: float
    state_gate: torch.Tensor
    state_contribution_norm: float
    rule_attention: torch.Tensor
    rule_contribution_norm: float
    residual_scale: float
    adapter_enabled: bool
    memory_delta_norm: float = 0.0
    rule_delta_norm: float = 0.0
    state_delta_norm: float = 0.0
    base_delta_norm: float = 0.0
    memory_residual_scale: float = 0.0
    rule_residual_scale: float = 0.0
    path_dominance_ratio: float = 0.0
    path_specific_adapter_version: str = "baseline_v1"
    memory_delta_tensor: torch.Tensor | None = None
    rule_delta_tensor: torch.Tensor | None = None
    state_delta_tensor: torch.Tensor | None = None
    base_delta_tensor: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance.to(hidden.dtype) + self.eps) * self.weight


class CivilizationAdapter(nn.Module):
    def __init__(self, config: CivilizationAdapterConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        bottleneck = config.bottleneck_size
        self.norm = RMSNorm(hidden)
        self.down_projection = nn.Linear(hidden, bottleneck, bias=False)
        self.memory_key = nn.Linear(hidden, bottleneck, bias=False)
        self.memory_value = nn.Linear(hidden, bottleneck, bias=False)
        self.rule_key = nn.Linear(hidden, bottleneck, bias=False)
        self.rule_value = nn.Linear(hidden, bottleneck, bias=False)
        self.state_projection = nn.Linear(3, bottleneck, bias=False)
        self.state_gates = nn.Linear(3, 3)
        self.up_projection = nn.Linear(bottleneck, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
        self.last_trace: CivilizationAdapterTrace | None = None
        self.last_pre_scale_update: torch.Tensor | None = None

    @staticmethod
    def _cross_attention(
        query: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, bottleneck = query.shape
        if vectors.shape[1] == 0:
            empty = query.new_zeros((batch, seq_len, 0))
            return torch.zeros_like(query), empty
        keys = key_projection(vectors)
        values = value_projection(vectors)
        scores = torch.matmul(query, keys.transpose(-1, -2)) / (bottleneck**0.5)
        valid = mask.to(scores.device, dtype=torch.bool).unsqueeze(1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid.to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
        attention = attention / denominator
        return torch.matmul(attention, values), attention

    def forward(
        self,
        hidden: torch.Tensor,
        context: CivilizationAdapterContext,
    ) -> tuple[torch.Tensor, CivilizationAdapterTrace]:
        ablation = context.ablation_config or CivilizationAblationConfig()
        pre_norm = torch.linalg.vector_norm(hidden.float(), dim=-1).mean()
        if not context.adapter_enabled:
            self.last_pre_scale_update = torch.zeros_like(hidden)
            trace = self._zero_trace(hidden, pre_norm, context, enabled=False)
            self.last_trace = trace
            return hidden, trace

        normalized = self.norm(hidden)
        bottleneck = self.down_projection(normalized)
        state_values = context.state_values.to(hidden.device, hidden.dtype)
        gates = (
            torch.sigmoid(self.state_gates(state_values))
            if self.config.use_state_path and ablation.use_state_path
            else torch.ones((hidden.shape[0], 3), dtype=hidden.dtype, device=hidden.device)
        )

        memory_context = torch.zeros_like(bottleneck)
        memory_attention = hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0))
        if self.config.use_memory_path and ablation.use_memory_path:
            memory_context, memory_attention = self._cross_attention(
                bottleneck,
                context.memory_vectors.to(hidden.device, hidden.dtype),
                context.memory_mask,
                self.memory_key,
                self.memory_value,
            )
            memory_context = memory_context * gates[:, 0].view(-1, 1, 1)

        state_context = torch.zeros_like(bottleneck)
        if self.config.use_state_path and ablation.use_state_path:
            state_context = self.state_projection(state_values).unsqueeze(1)

        rule_context = torch.zeros_like(bottleneck)
        rule_attention = hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0))
        if self.config.use_rule_path and ablation.use_rule_path:
            rule_context, rule_attention = self._cross_attention(
                bottleneck,
                context.rule_vectors.to(hidden.device, hidden.dtype),
                context.rule_mask,
                self.rule_key,
                self.rule_value,
            )
            rule_context = rule_context * gates[:, 1].view(-1, 1, 1)

        update = F.silu(bottleneck + memory_context + state_context + rule_context)
        update = self.up_projection(self.dropout(update))
        self.last_pre_scale_update = update
        state_scale = gates[:, 2].view(-1, 1, 1) if ablation.use_state_path else 1.0
        scale = torch.zeros_like(self.residual_scale) if context.force_zero_scale else self.residual_scale
        delta = update * state_scale * scale
        output = hidden + delta
        if not torch.isfinite(output).all():
            raise ValueError("CivilizationAdapter produced NaN or Inf")

        post_norm = torch.linalg.vector_norm(output.float(), dim=-1).mean()
        ratio = post_norm / pre_norm.clamp_min(1e-8)
        trace = CivilizationAdapterTrace(
            target_layer=self.config.target_layer,
            pre_hidden_norm=float(pre_norm.detach().cpu()),
            post_hidden_norm=float(post_norm.detach().cpu()),
            delta_norm=float(torch.linalg.vector_norm(delta.float(), dim=-1).mean().detach().cpu()),
            hidden_norm_ratio=float(ratio.detach().cpu()),
            memory_attention=memory_attention.detach().cpu(),
            memory_contribution_norm=float(torch.linalg.vector_norm(memory_context.float(), dim=-1).mean().detach().cpu()),
            state_gate=gates.detach().cpu(),
            state_contribution_norm=float(torch.linalg.vector_norm(state_context.float(), dim=-1).mean().detach().cpu()),
            rule_attention=rule_attention.detach().cpu(),
            rule_contribution_norm=float(torch.linalg.vector_norm(rule_context.float(), dim=-1).mean().detach().cpu()),
            residual_scale=float(scale.detach().cpu()),
            adapter_enabled=True,
        )
        if trace.hidden_norm_ratio > 2.0:
            raise ValueError(f"Adapter hidden norm ratio exceeded limit: {trace.hidden_norm_ratio}")
        self.last_trace = trace
        return output, trace

    def _zero_trace(
        self,
        hidden: torch.Tensor,
        pre_norm: torch.Tensor,
        context: CivilizationAdapterContext,
        enabled: bool,
    ) -> CivilizationAdapterTrace:
        return CivilizationAdapterTrace(
            target_layer=self.config.target_layer,
            pre_hidden_norm=float(pre_norm.detach().cpu()),
            post_hidden_norm=float(pre_norm.detach().cpu()),
            delta_norm=0.0,
            hidden_norm_ratio=1.0,
            memory_attention=hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0)).cpu(),
            memory_contribution_norm=0.0,
            state_gate=hidden.new_zeros((hidden.shape[0], 3)).cpu(),
            state_contribution_norm=0.0,
            rule_attention=hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0)).cpu(),
            rule_contribution_norm=0.0,
            residual_scale=0.0,
            adapter_enabled=enabled,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class PathSpecificCivilizationAdapter(nn.Module):
    """Adapter variant with independently scaled Memory/Rule/State/Base residual paths."""

    def __init__(self, config: CivilizationAdapterConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        bottleneck = config.bottleneck_size
        self.norm = RMSNorm(hidden)
        self.down_projection = nn.Linear(hidden, bottleneck, bias=False)
        self.base_up_projection = nn.Linear(bottleneck, hidden, bias=False)

        self.memory_learned_query = nn.Parameter(torch.zeros(1, 1, bottleneck))
        self.memory_query_projection = nn.Linear(bottleneck, bottleneck, bias=False)
        self.memory_key = nn.Linear(hidden, bottleneck, bias=False)
        self.memory_value = nn.Linear(hidden, bottleneck, bias=False)
        self.memory_up_projection = nn.Linear(bottleneck, hidden, bias=False)

        self.rule_learned_query = nn.Parameter(torch.zeros(1, 1, bottleneck))
        self.rule_query_projection = nn.Linear(bottleneck, bottleneck, bias=False)
        self.rule_key = nn.Linear(hidden, bottleneck, bias=False)
        self.rule_value = nn.Linear(hidden, bottleneck, bias=False)
        self.rule_up_projection = nn.Linear(bottleneck, hidden, bias=False)

        self.state_projection = nn.Linear(3, bottleneck, bias=False)
        self.state_gates = nn.Linear(3, 4)
        self.state_up_projection = nn.Linear(bottleneck, hidden, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        init = float(config.residual_scale_init)
        self.base_residual_scale = nn.Parameter(torch.tensor(init))
        self.memory_residual_scale = nn.Parameter(torch.tensor(init))
        self.rule_residual_scale = nn.Parameter(torch.tensor(init))
        self.state_residual_scale = nn.Parameter(torch.tensor(init))
        self.last_trace: CivilizationAdapterTrace | None = None
        self.last_pre_scale_update: torch.Tensor | None = None

    @property
    def residual_scale(self) -> torch.Tensor:
        # Compatibility for callers that expect a single scale.
        return self.base_residual_scale

    @staticmethod
    def _cross_attention(
        query: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, bottleneck = query.shape
        if vectors.shape[1] == 0:
            empty = query.new_zeros((batch, seq_len, 0))
            return torch.zeros_like(query), empty
        keys = key_projection(vectors)
        values = value_projection(vectors)
        scores = torch.matmul(query, keys.transpose(-1, -2)) / (bottleneck**0.5)
        valid = mask.to(scores.device, dtype=torch.bool).unsqueeze(1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid.to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
        attention = attention / denominator
        return torch.matmul(attention, values), attention

    def forward(
        self,
        hidden: torch.Tensor,
        context: CivilizationAdapterContext,
    ) -> tuple[torch.Tensor, CivilizationAdapterTrace]:
        ablation = context.ablation_config or CivilizationAblationConfig()
        pre_norm = torch.linalg.vector_norm(hidden.float(), dim=-1).mean()
        if not context.adapter_enabled:
            self.last_pre_scale_update = torch.zeros_like(hidden)
            trace = self._zero_trace(hidden, pre_norm, context, enabled=False)
            self.last_trace = trace
            return hidden, trace

        normalized = self.norm(hidden)
        bottleneck = self.down_projection(normalized)
        state_values = context.state_values.to(hidden.device, hidden.dtype)
        gates = (
            torch.sigmoid(self.state_gates(state_values))
            if self.config.use_state_path and ablation.use_state_path
            else torch.ones((hidden.shape[0], 4), dtype=hidden.dtype, device=hidden.device)
        )

        base_update = self.base_up_projection(self.dropout(F.silu(bottleneck)))
        memory_context = torch.zeros_like(bottleneck)
        memory_attention = hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0))
        if self.config.use_memory_path and ablation.use_memory_path:
            memory_query = self.memory_query_projection(bottleneck) + self.memory_learned_query.to(hidden.dtype)
            memory_context, memory_attention = self._cross_attention(
                memory_query,
                context.memory_vectors.to(hidden.device, hidden.dtype),
                context.memory_mask,
                self.memory_key,
                self.memory_value,
            )
            memory_context = memory_context * gates[:, 0].view(-1, 1, 1)
        memory_update = self.memory_up_projection(self.dropout(F.silu(memory_context)))

        rule_context = torch.zeros_like(bottleneck)
        rule_attention = hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0))
        if self.config.use_rule_path and ablation.use_rule_path:
            rule_query = self.rule_query_projection(bottleneck) + self.rule_learned_query.to(hidden.dtype)
            rule_context, rule_attention = self._cross_attention(
                rule_query,
                context.rule_vectors.to(hidden.device, hidden.dtype),
                context.rule_mask,
                self.rule_key,
                self.rule_value,
            )
            rule_context = rule_context * gates[:, 1].view(-1, 1, 1)
        rule_update = self.rule_up_projection(self.dropout(F.silu(rule_context)))

        state_context = torch.zeros_like(bottleneck)
        if self.config.use_state_path and ablation.use_state_path:
            state_context = self.state_projection(state_values).unsqueeze(1)
        state_update = self.state_up_projection(self.dropout(F.silu(state_context)))

        zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        base_scale = zero if context.force_zero_scale else self.base_residual_scale
        memory_scale = zero if context.force_zero_scale else self.memory_residual_scale
        rule_scale = zero if context.force_zero_scale else self.rule_residual_scale
        state_scale = zero if context.force_zero_scale else self.state_residual_scale

        base_delta = base_update * gates[:, 3].view(-1, 1, 1) * base_scale
        memory_delta = memory_update * memory_scale if self.config.use_memory_path and ablation.use_memory_path else torch.zeros_like(hidden)
        rule_delta = rule_update * rule_scale if self.config.use_rule_path and ablation.use_rule_path else torch.zeros_like(hidden)
        state_delta = state_update * state_scale if self.config.use_state_path and ablation.use_state_path else torch.zeros_like(hidden)
        delta = base_delta + memory_delta + rule_delta + state_delta
        output = hidden + delta
        self.last_pre_scale_update = base_update + memory_update + rule_update + state_update
        if not torch.isfinite(output).all():
            raise ValueError("PathSpecificCivilizationAdapter produced NaN or Inf")

        post_norm = torch.linalg.vector_norm(output.float(), dim=-1).mean()
        ratio = post_norm / pre_norm.clamp_min(1e-8)
        base_delta_norm = torch.linalg.vector_norm(base_delta.float(), dim=-1).mean()
        memory_delta_norm = torch.linalg.vector_norm(memory_delta.float(), dim=-1).mean()
        rule_delta_norm = torch.linalg.vector_norm(rule_delta.float(), dim=-1).mean()
        state_delta_norm = torch.linalg.vector_norm(state_delta.float(), dim=-1).mean()
        path_norms = torch.stack([base_delta_norm, memory_delta_norm, rule_delta_norm, state_delta_norm])
        path_dominance = path_norms.max() / path_norms.sum().clamp_min(1e-8)
        trace = CivilizationAdapterTrace(
            target_layer=self.config.target_layer,
            pre_hidden_norm=float(pre_norm.detach().cpu()),
            post_hidden_norm=float(post_norm.detach().cpu()),
            delta_norm=float(torch.linalg.vector_norm(delta.float(), dim=-1).mean().detach().cpu()),
            hidden_norm_ratio=float(ratio.detach().cpu()),
            memory_attention=memory_attention.detach().cpu(),
            memory_contribution_norm=float(torch.linalg.vector_norm(memory_context.float(), dim=-1).mean().detach().cpu()),
            state_gate=gates[:, :3].detach().cpu(),
            state_contribution_norm=float(torch.linalg.vector_norm(state_context.float(), dim=-1).mean().detach().cpu()),
            rule_attention=rule_attention.detach().cpu(),
            rule_contribution_norm=float(torch.linalg.vector_norm(rule_context.float(), dim=-1).mean().detach().cpu()),
            residual_scale=float(base_scale.detach().cpu()),
            adapter_enabled=True,
            memory_delta_norm=float(memory_delta_norm.detach().cpu()),
            rule_delta_norm=float(rule_delta_norm.detach().cpu()),
            state_delta_norm=float(state_delta_norm.detach().cpu()),
            base_delta_norm=float(base_delta_norm.detach().cpu()),
            memory_residual_scale=float(memory_scale.detach().cpu()),
            rule_residual_scale=float(rule_scale.detach().cpu()),
            path_dominance_ratio=float(path_dominance.detach().cpu()),
            path_specific_adapter_version="path_specific_v2",
            memory_delta_tensor=memory_delta,
            rule_delta_tensor=rule_delta,
            state_delta_tensor=state_delta,
            base_delta_tensor=base_delta,
        )
        if trace.hidden_norm_ratio > 2.0:
            raise ValueError(f"Path-specific adapter hidden norm ratio exceeded limit: {trace.hidden_norm_ratio}")
        self.last_trace = trace
        return output, trace

    def _zero_trace(
        self,
        hidden: torch.Tensor,
        pre_norm: torch.Tensor,
        context: CivilizationAdapterContext,
        enabled: bool,
    ) -> CivilizationAdapterTrace:
        return CivilizationAdapterTrace(
            target_layer=self.config.target_layer,
            pre_hidden_norm=float(pre_norm.detach().cpu()),
            post_hidden_norm=float(pre_norm.detach().cpu()),
            delta_norm=0.0,
            hidden_norm_ratio=1.0,
            memory_attention=hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0)).cpu(),
            memory_contribution_norm=0.0,
            state_gate=hidden.new_zeros((hidden.shape[0], 3)).cpu(),
            state_contribution_norm=0.0,
            rule_attention=hidden.new_zeros((hidden.shape[0], hidden.shape[1], 0)).cpu(),
            rule_contribution_norm=0.0,
            residual_scale=0.0,
            adapter_enabled=enabled,
            path_specific_adapter_version="path_specific_v2",
            memory_delta_tensor=torch.zeros_like(hidden),
            rule_delta_tensor=torch.zeros_like(hidden),
            state_delta_tensor=torch.zeros_like(hidden),
            base_delta_tensor=torch.zeros_like(hidden),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
