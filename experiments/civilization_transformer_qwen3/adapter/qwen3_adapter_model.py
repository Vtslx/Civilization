from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..backend import Qwen3Backend
from .civilization_adapter import CivilizationAdapter, CivilizationAdapterContext, CivilizationAdapterTrace


@dataclass
class Qwen3AdapterOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    trace: CivilizationAdapterTrace
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class Qwen3AdapterModel(nn.Module):
    def __init__(self, backend: Qwen3Backend, adapter: CivilizationAdapter):
        super().__init__()
        self.backend = backend
        self.adapter = adapter.to(backend.device, dtype=backend.dtype)
        self.active_hook_count = 0

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        context: CivilizationAdapterContext,
    ) -> Qwen3AdapterOutput:
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        target_layer = self.adapter.config.target_layer
        if not 0 <= target_layer < len(self.backend.model.model.layers):
            raise ValueError(f"target_layer out of range: {target_layer}")
        trace_holder: dict[str, Any] = {}

        def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
            injected, trace = self.adapter(output, context)
            trace_holder["trace"] = trace
            return injected

        handle = self.backend.model.model.layers[target_layer].register_forward_hook(hook)
        self.active_hook_count += 1
        try:
            output = self.backend.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
            self.active_hook_count -= 1
        if self.active_hook_count != 0:
            raise RuntimeError("Qwen3 adapter hook leaked after forward")
        if "trace" not in trace_holder:
            raise RuntimeError("Qwen3 adapter hook did not execute")
        if not torch.isfinite(output.logits).all():
            raise ValueError("Qwen3 adapter logits contain NaN or Inf")
        return Qwen3AdapterOutput(
            logits=output.logits,
            hidden_states=tuple(output.hidden_states),
            trace=trace_holder["trace"],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.adapter.parameters())

    def generate(
        self,
        encoded: dict[str, torch.Tensor],
        context: CivilizationAdapterContext,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        target_layer = self.adapter.config.target_layer

        def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
            injected, _trace = self.adapter(output, context)
            return injected

        handle = self.backend.model.model.layers[target_layer].register_forward_hook(hook)
        self.active_hook_count += 1
        try:
            with torch.no_grad():
                generated = self.backend.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                )
        finally:
            handle.remove()
            self.active_hook_count -= 1
        if self.active_hook_count != 0:
            raise RuntimeError("Qwen3 adapter hook leaked after generation")
        return generated
