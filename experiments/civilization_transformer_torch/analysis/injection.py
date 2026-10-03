from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dataset import LOGIC_LABELS


@dataclass(frozen=True)
class InjectionTrace:
    target_label: str
    alpha: float
    layer_index: int
    strategy: str
    position: str
    pre_norm: float
    post_norm: float
    delta_norm: float
    hidden_norm_ratio: float
    warning: str | None

    def to_dict(self) -> dict:
        return {
            "target_label": self.target_label,
            "alpha": self.alpha,
            "layer_index": self.layer_index,
            "strategy": self.strategy,
            "position": self.position,
            "pre_norm": self.pre_norm,
            "post_norm": self.post_norm,
            "delta_norm": self.delta_norm,
            "hidden_norm_ratio": self.hidden_norm_ratio,
            "warning": self.warning,
        }


class LogicCodeInjector:
    def __init__(
        self,
        centroids: dict[str, np.ndarray | torch.Tensor],
        target_label: str,
        model_dim: int,
        layer_index: int,
        alpha: float = 0.25,
        strategy: str = "residual_norm",
        position: str = "all",
        max_norm_ratio: float = 2.0,
    ):
        if target_label not in LOGIC_LABELS:
            raise ValueError(f"unknown target_label {target_label}")
        if target_label not in centroids:
            raise ValueError(f"missing centroid for target_label {target_label}")
        if strategy not in {"additive", "gated", "residual_norm"}:
            raise ValueError("strategy must be additive, gated, or residual_norm")
        if position not in {"all", "last_token", "first_token"}:
            raise ValueError("position must be all, last_token, or first_token")
        vector = torch.as_tensor(centroids[target_label], dtype=torch.float32)
        if vector.ndim != 1 or vector.shape[0] != model_dim:
            raise ValueError("injection vector shape must be [model_dim]")
        self.centroids = centroids
        self.target_label = target_label
        self.model_dim = model_dim
        self.layer_index = layer_index
        self.alpha = float(alpha)
        self.strategy = strategy
        self.position = position
        self.max_norm_ratio = float(max_norm_ratio)
        self.vector = vector

    def __call__(self, hidden: torch.Tensor, layer_index: int) -> tuple[torch.Tensor, dict | None]:
        if layer_index != self.layer_index:
            return hidden, None
        if hidden.ndim != 3 or hidden.shape[-1] != self.model_dim:
            raise ValueError("hidden must have shape [batch, seq, model_dim]")
        if self.alpha == 0.0:
            trace = self._trace(hidden, hidden, torch.zeros_like(hidden), layer_index, None)
            return hidden, trace.to_dict()

        vector = self.vector.to(hidden.device, hidden.dtype)
        delta = torch.zeros_like(hidden)
        if self.position == "all":
            delta = vector.view(1, 1, -1).expand_as(hidden)
        elif self.position == "last_token":
            delta[:, -1, :] = vector
        else:
            delta[:, 0, :] = vector

        if self.strategy == "gated":
            gate = torch.sigmoid(hidden.mean(dim=-1, keepdim=True))
            delta = delta * gate

        proposed = hidden + self.alpha * delta
        if self.strategy == "residual_norm":
            proposed = self._norm_clamp(hidden, proposed)

        applied_delta = proposed - hidden
        warning = None
        ratio = self._norm_ratio(hidden, proposed)
        if ratio > self.max_norm_ratio:
            warning = "hidden_norm_ratio_exceeded"
        trace = self._trace(hidden, proposed, applied_delta, layer_index, warning)
        return proposed, trace.to_dict()

    def _norm_clamp(self, hidden: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        pre_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True).clamp_min(1e-8)
        post_norm = torch.linalg.vector_norm(proposed, dim=-1, keepdim=True).clamp_min(1e-8)
        max_allowed = pre_norm * self.max_norm_ratio
        scale = torch.minimum(torch.ones_like(post_norm), max_allowed / post_norm)
        return hidden + (proposed - hidden) * scale

    def _norm_ratio(self, hidden: torch.Tensor, proposed: torch.Tensor) -> float:
        pre = torch.linalg.vector_norm(hidden).detach().cpu().item()
        post = torch.linalg.vector_norm(proposed).detach().cpu().item()
        return float(post / max(pre, 1e-8))

    def _trace(self, hidden: torch.Tensor, proposed: torch.Tensor, delta: torch.Tensor, layer_index: int, warning: str | None) -> InjectionTrace:
        pre = float(torch.linalg.vector_norm(hidden).detach().cpu().item())
        post = float(torch.linalg.vector_norm(proposed).detach().cpu().item())
        ratio = float(post / max(pre, 1e-8))
        if warning is None and ratio > self.max_norm_ratio:
            warning = "hidden_norm_ratio_exceeded"
        if warning is None and self.alpha >= 5.0:
            warning = "alpha_too_high"
        return InjectionTrace(
            target_label=self.target_label,
            alpha=self.alpha,
            layer_index=layer_index,
            strategy=self.strategy,
            position=self.position,
            pre_norm=pre,
            post_norm=post,
            delta_norm=float(torch.linalg.vector_norm(delta).detach().cpu().item()),
            hidden_norm_ratio=ratio,
            warning=warning,
        )


def nearest_centroid_label(vector: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[str, float]:
    if vector.ndim != 1:
        raise ValueError("vector must have shape [model_dim]")
    distances = {label: float(np.linalg.norm(vector - centroid)) for label, centroid in centroids.items()}
    label = min(distances, key=distances.get)
    return label, distances[label]
