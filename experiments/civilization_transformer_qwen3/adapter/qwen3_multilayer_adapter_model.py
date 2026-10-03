from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..backend import Qwen3Backend
from .civilization_adapter import CivilizationAdapter, CivilizationAdapterContext, CivilizationAdapterTrace


@dataclass
class Qwen3MultiAdapterOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    traces: dict[int, CivilizationAdapterTrace]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class Qwen3MultiAdapterModel(nn.Module):
    def __init__(self, backend: Qwen3Backend, adapters: dict[int, CivilizationAdapter]):
        super().__init__()
        if not adapters:
            raise ValueError("at least one adapter is required")
        self.backend = backend
        self.adapters = nn.ModuleDict(
            {
                str(layer): adapter.to(backend.device, dtype=backend.dtype)
                for layer, adapter in sorted(adapters.items())
            }
        )
        self.active_hook_count = 0

    @property
    def target_layers(self) -> tuple[int, ...]:
        return tuple(int(layer) for layer in self.adapters.keys())

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        context: CivilizationAdapterContext,
    ) -> Qwen3MultiAdapterOutput:
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        trace_holder: dict[int, CivilizationAdapterTrace] = {}
        handles = []

        def make_hook(layer: int):
            def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
                injected, trace = self.adapters[str(layer)](output, context)
                trace_holder[layer] = trace
                return injected

            return hook

        try:
            for layer in self.target_layers:
                if not 0 <= layer < len(self.backend.model.model.layers):
                    raise ValueError(f"target_layer out of range: {layer}")
                handles.append(self.backend.model.model.layers[layer].register_forward_hook(make_hook(layer)))
                self.active_hook_count += 1
            output = self.backend.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
                self.active_hook_count -= 1
        if self.active_hook_count != 0:
            raise RuntimeError("Qwen3 multi-adapter hook leaked after forward")
        if set(trace_holder) != set(self.target_layers):
            raise RuntimeError("not all Qwen3 multi-adapter hooks executed")
        if not torch.isfinite(output.logits).all():
            raise ValueError("Qwen3 multi-adapter logits contain NaN or Inf")
        return Qwen3MultiAdapterOutput(
            logits=output.logits,
            hidden_states=tuple(output.hidden_states),
            traces=trace_holder,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters())

    def generate(
        self,
        encoded: dict[str, torch.Tensor],
        context: CivilizationAdapterContext,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        input_ids = encoded["input_ids"].to(self.backend.device)
        attention_mask = encoded["attention_mask"].to(self.backend.device)
        handles = []

        def make_hook(layer: int):
            def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
                injected, _trace = self.adapters[str(layer)](output, context)
                return injected

            return hook

        try:
            for layer in self.target_layers:
                handles.append(self.backend.model.model.layers[layer].register_forward_hook(make_hook(layer)))
                self.active_hook_count += 1
            with torch.no_grad():
                generated = self.backend.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
                self.active_hook_count -= 1
        if self.active_hook_count != 0:
            raise RuntimeError("Qwen3 multi-adapter hook leaked after generation")
        return generated
