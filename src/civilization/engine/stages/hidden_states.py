from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from civilization.research.torch_line.analysis.dataset import LogicSample

from ..backend import Qwen3Backend


@dataclass
class Qwen3RepresentationResult:
    representations: dict[str, dict[int, torch.Tensor]]
    truncations: list[dict[str, Any]]
    resource_usage: list[dict[str, Any]]
    num_samples: int
    selected_layers: tuple[int, ...]


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden and attention_mask must have shapes [batch, seq, dim] and [batch, seq]")
    mask = attention_mask.to(hidden.device, hidden.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("attention mask contains an empty sequence")
    return (hidden * mask).sum(dim=1) / counts


def last_non_padding_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden and attention_mask must have shapes [batch, seq, dim] and [batch, seq]")
    lengths = attention_mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("attention mask contains an empty sequence")
    indices = (lengths - 1).to(hidden.device)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def collect_qwen3_hidden_states(
    backend: Qwen3Backend,
    samples: list[LogicSample],
    max_length: int = 128,
    batch_size: int = 1,
    pooling: tuple[str, ...] = ("mean", "last"),
    selected_layers: tuple[int, ...] | None = None,
) -> Qwen3RepresentationResult:
    if batch_size not in {1, 2}:
        raise ValueError("batch_size must be 1 or 2 for the 16GB migration baseline")
    if not samples:
        raise ValueError("samples cannot be empty")
    layers = tuple(range(29)) if selected_layers is None else tuple(selected_layers)
    if any(layer < 0 or layer > 28 for layer in layers):
        raise ValueError("selected_layers must be between 0 and 28")
    unsupported = set(pooling) - {"mean", "last"}
    if unsupported:
        raise ValueError(f"unsupported pooling modes: {sorted(unsupported)}")

    collected = {mode: {layer: [] for layer in layers} for mode in pooling}
    truncations: list[dict[str, Any]] = []
    resource_usage: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        started = time.perf_counter()
        output, batch_truncations = backend.forward_texts([sample.text for sample in batch], max_length=max_length)
        for truncation in batch_truncations:
            local_index = truncation["sample_index"]
            truncation.update(
                {
                    "sample_index": start + local_index,
                    "label": batch[local_index].label,
                    "variant": batch[local_index].variant,
                }
            )
        truncations.extend(batch_truncations)
        for layer in layers:
            hidden = output.hidden_states[layer]
            if "mean" in pooling:
                collected["mean"][layer].append(masked_mean_pool(hidden, output.attention_mask).float().cpu())
            if "last" in pooling:
                collected["last"][layer].append(last_non_padding_pool(hidden, output.attention_mask).float().cpu())
        resource_usage.append(
            {
                "batch_start": start,
                "batch_size": len(batch),
                "sequence_length": int(output.input_ids.shape[1]),
                "total_seconds": time.perf_counter() - started,
                **output.runtime_trace,
            }
        )
        del output
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    representations = {
        mode: {layer: torch.cat(parts, dim=0) for layer, parts in layer_data.items()}
        for mode, layer_data in collected.items()
    }
    for layer_data in representations.values():
        for vectors in layer_data.values():
            if vectors.shape != (len(samples), 1024):
                raise ValueError(f"unexpected representation shape: {tuple(vectors.shape)}")
            if not torch.isfinite(vectors).all():
                raise ValueError("pooled representations contain NaN or Inf")
    return Qwen3RepresentationResult(
        representations=representations,
        truncations=truncations,
        resource_usage=resource_usage,
        num_samples=len(samples),
        selected_layers=layers,
    )
