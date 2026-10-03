from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any

import psutil
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DynamicCache

from ..backend import Qwen3Backend
from .evidence_answer_training import _answer_scores
from .group_full_hidden_centroid_integration import _centroid_scores
from .hidden_states import last_non_padding_pool
from .matched_baseline_benchmark import (
    CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
    DEFAULT_OPTIMIZATION_STEPS,
    _build_centroids,
    _evaluate,
    _group_texts,
    _json_dump,
    _option_vectors,
    _seed_everything,
    _sha,
    _targets,
    _write_csv,
)
from ..adapter.context_encoder import qwen_text_for_sample
from .rule_conflict_group_recovery import _build_group_splits


METHOD_LORA = "lora"
METHOD_PREFIX = "prefix_tuning"
SUPPORTED_PEFT_METHODS = (METHOD_LORA, METHOD_PREFIX)
LORA_RANK = 13
LORA_ALPHA = 26.0
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
PREFIX_LENGTH = 65


class LoRAUpdate(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(inputs)) * self.scaling


class Qwen3LoRAModel(nn.Module):
    """LoRA updates held outside Qwen so the frozen-base audit remains exact."""

    def __init__(self, backend: Qwen3Backend, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> None:
        super().__init__()
        self.backend = backend
        updates = {}
        self.targets = []
        for layer_index, layer in enumerate(backend.model.model.layers):
            for target_name in LORA_TARGETS:
                target = getattr(layer.self_attn, target_name)
                key = f"layer_{layer_index}_{target_name}"
                updates[key] = LoRAUpdate(target.in_features, target.out_features, rank, alpha).to(
                    backend.device, dtype=backend.dtype
                )
                self.targets.append((key, target))
        self.updates = nn.ModuleDict(updates)
        self.active_hook_count = 0

    def forward(self, encoded: dict[str, torch.Tensor]):
        handles = []

        def make_hook(key: str):
            def hook(_module, args, output):
                return output + self.updates[key](args[0])
            return hook

        try:
            for key, target in self.targets:
                handles.append(target.register_forward_hook(make_hook(key)))
                self.active_hook_count += 1
            # The benchmark consumes hidden states only. Calling the decoder
            # directly avoids materializing [batch, seq, 151936] LM logits.
            output = self.backend.model.model(
                input_ids=encoded["input_ids"].to(self.backend.device),
                attention_mask=encoded["attention_mask"].to(self.backend.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
                self.active_hook_count -= 1
        if self.active_hook_count:
            raise RuntimeError("LoRA hook leaked")
        return output, encoded["attention_mask"].to(self.backend.device)


class Qwen3PrefixTuningModel(nn.Module):
    """Direct deep K/V prefix tuning for every Qwen3 transformer layer."""

    def __init__(self, backend: Qwen3Backend, prefix_length: int = PREFIX_LENGTH) -> None:
        super().__init__()
        self.backend = backend
        self.prefix_length = prefix_length
        config = backend.config
        shape = (
            config.num_hidden_layers,
            2,
            config.num_key_value_heads,
            prefix_length,
            config.head_dim,
        )
        prefix = torch.empty(shape, device=backend.device, dtype=backend.dtype)
        nn.init.normal_(prefix, mean=0.0, std=0.02)
        self.prefix_key_values = nn.Parameter(prefix)

    def _cache(self, batch_size: int) -> DynamicCache:
        cache = DynamicCache()
        for layer_index in range(self.prefix_key_values.shape[0]):
            keys = self.prefix_key_values[layer_index, 0].unsqueeze(0).expand(batch_size, -1, -1, -1)
            values = self.prefix_key_values[layer_index, 1].unsqueeze(0).expand(batch_size, -1, -1, -1)
            cache.update(keys, values, layer_index)
        return cache

    def forward(self, encoded: dict[str, torch.Tensor]):
        input_ids = encoded["input_ids"].to(self.backend.device)
        input_mask = encoded["attention_mask"].to(self.backend.device)
        batch_size, sequence_length = input_ids.shape
        prefix_mask = torch.ones(
            (batch_size, self.prefix_length),
            dtype=input_mask.dtype,
            device=self.backend.device,
        )
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)
        cache_position = torch.arange(
            self.prefix_length,
            self.prefix_length + sequence_length,
            dtype=torch.long,
            device=self.backend.device,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        output = self.backend.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._cache(batch_size),
            use_cache=True,
            cache_position=cache_position,
            output_hidden_states=True,
            return_dict=True,
        )
        return output, input_mask


def lora_parameter_count(config, rank: int = LORA_RANK) -> int:
    q = rank * (config.hidden_size + config.num_attention_heads * config.head_dim)
    kv = rank * (config.hidden_size + config.num_key_value_heads * config.head_dim)
    o = rank * (config.num_attention_heads * config.head_dim + config.hidden_size)
    return config.num_hidden_layers * (q + kv + kv + o)


def prefix_parameter_count(config, prefix_length: int = PREFIX_LENGTH) -> int:
    return (
        config.num_hidden_layers
        * 2
        * config.num_key_value_heads
        * prefix_length
        * config.head_dim
    )


def _pooled_forward(backend, model, texts: list[str], max_length: int):
    tokenized = backend.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    counts = [len(row) for row in tokenized["input_ids"]]
    if max(counts, default=0) > max_length:
        raise ValueError(f"PEFT baseline input truncation: max_tokens={max(counts)} max_length={max_length}")
    encoded, truncations = backend.encode(texts, max_length=max_length)
    if truncations:
        raise ValueError("PEFT baseline input was truncated")
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    output, mask = model(encoded)
    return last_non_padding_pool(output.hidden_states[-1], mask).float(), counts


def _balanced_groups(groups, step: int, seed: int):
    result = []
    for offset, group_type in enumerate(("memory_necessity_group", "rule_necessity_group", "memory_rule_conflict_group")):
        rows = sorted((group for group in groups if group.group_type == group_type), key=lambda group: group.surface_group_id)
        result.append(rows[(step + seed + offset) % len(rows)])
    return result


def _train(*, backend, model, train_groups, steps: int, seed: int, max_length: int, method: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rows = []
    for step in range(steps):
        groups = _balanced_groups(train_groups, step, seed)
        full_texts = [text for group in groups for text in _group_texts(group, "full", include_context=True)]
        wrong_texts = [text for group in groups for text in _group_texts(group, "wrong_context", include_context=True)]
        targets = torch.cat([_targets(group, backend.device) for group in groups])
        optimizer.zero_grad(set_to_none=True)
        pooled, counts = _pooled_forward(backend, model, full_texts + wrong_texts, max_length)
        full = pooled[: targets.numel()]
        wrong = pooled[targets.numel() :]
        options = _option_vectors(backend, groups[0])
        full_scores = _answer_scores(full, options)
        wrong_scores = _answer_scores(wrong, options)
        centroids = torch.stack([full[targets == option].mean(dim=0) for option in range(5)])
        dynamic_scores = _centroid_scores(full, centroids)
        correct = full_scores.gather(1, targets[:, None]).squeeze(1)
        wrong_correct = wrong_scores.gather(1, targets[:, None]).squeeze(1)
        loss = (
            F.cross_entropy(full_scores / 0.05, targets)
            + 2.0 * F.cross_entropy(dynamic_scores / 0.05, targets)
            + torch.relu(0.20 - (correct - wrong_correct)).mean()
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "train_accuracy": float((full_scores.argmax(-1) == targets).float().mean().detach().cpu()),
            "gradient_norm": grad_norm,
            "max_input_tokens": max(counts),
        }
        rows.append(row)
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            print(
                f"matched_baseline method={method} seed={seed} step={step + 1}/{steps} "
                f"loss={row['loss']:.6f} accuracy={row['train_accuracy']:.3f}",
                flush=True,
            )
    return rows


def run_peft_matched_baseline_seed(
    *,
    backend: Qwen3Backend,
    output_dir: str | Path,
    method: str,
    seed: int,
    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS,
    local_samples_per_label: int = 16,
    local_train_groups: int = 10,
    max_length: int = 160,
) -> dict[str, Any]:
    if method not in SUPPORTED_PEFT_METHODS:
        raise ValueError(f"unsupported PEFT method: {method}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    train, heldout, manifest = _build_group_splits(
        local_samples_per_label=local_samples_per_label,
        local_train_groups=local_train_groups,
        seed=seed,
        max_length=64,
    )
    sample_contract = [
        {
            "split": split,
            "group_type": group.group_type,
            "surface_group_id": group.surface_group_id,
            "targets": [pair.expected_full_option_id for pair in group.pairs],
            "task_text_sha256": _sha(qwen_text_for_sample(group.pairs[0].full_sample)),
        }
        for split, groups in (("train", train), ("heldout", heldout))
        for group in groups
    ]
    model = Qwen3LoRAModel(backend) if method == METHOD_LORA else Qwen3PrefixTuningModel(backend)
    initial_fingerprint = backend.parameter_fingerprint()
    gc.collect()
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    rss_before = psutil.Process().memory_info().rss
    losses = _train(
        backend=backend,
        model=model,
        train_groups=train,
        steps=optimization_steps,
        seed=seed,
        max_length=max_length,
        method=method,
    )
    # Imported helpers only require a model(encoded) -> (output, input_mask) contract.
    centroids, centroid_rows = _build_centroids(
        backend=backend,
        model=model,
        groups=train,
        include_context=True,
        max_length=max_length,
    )
    metric_rows, prompt_rows = _evaluate(
        backend=backend,
        model=model,
        groups=heldout,
        centroids=centroids,
        include_context=True,
        max_length=max_length,
    )
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
    runtime = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in model.parameters())
    values = lambda mode: [
        float(row["accuracy"]) for row in metric_rows if row["mode"] == mode
    ]
    mean = lambda mode: sum(values(mode)) / len(values(mode))
    full = mean("full")
    qwen_gradients = sum(
        1 for parameter in backend.model.parameters()
        if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu()) > 0
    )
    resource = {
        "runtime_seconds": runtime,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": psutil.Process().memory_info().rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(backend.device) if backend.device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(backend.device) if backend.device.type == "cuda" else 0,
    }
    summary = {
        "method": method,
        "seed": seed,
        "model": "Qwen3-0.6B",
        "model_sha256": backend.initial_sha256,
        "optimization_steps": optimization_steps,
        "trainable_parameter_count": trainable,
        "civilization_optimized_parameter_count": CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
        "parameter_match_relative_error": abs(trainable - CIVILIZATION_OPTIMIZED_PARAMETER_COUNT) / CIVILIZATION_OPTIMIZED_PARAMETER_COUNT,
        "accuracy": full,
        "wrong_context_accuracy": mean("wrong_context"),
        "wrong_context_drop": full - mean("wrong_context"),
        "memory_ablation_drop": full - mean("no_memory"),
        "rule_ablation_drop": full - mean("no_rule"),
        "state_ablation_drop": full - mean("no_state"),
        "empty_context_drop": full - mean("empty_context"),
        "ablation_semantics": "information_channel",
        "qwen_trainable_parameters": sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        "qwen_gradients": qwen_gradients,
        "qwen_weights_unchanged": initial_fingerprint == backend.parameter_fingerprint() and backend.verify_weights_unchanged(),
        **resource,
    }
    _json_dump(output / "dataset_manifest.json", manifest)
    _write_csv(output / "sample_contract.csv", sample_contract)
    _write_csv(output / "loss_curves.csv", losses)
    _write_csv(output / "centroid_build_audit.csv", centroid_rows)
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "prompt_audit.csv", prompt_rows)
    _json_dump(output / "resource_usage.json", resource)
    _json_dump(output / "summary.json", summary)
    torch.save(
        {
            "method_state_dict": model.state_dict(),
            "training_metadata": {
                "method": method,
                "seed": seed,
                "optimization_steps": optimization_steps,
                "lora_rank": LORA_RANK if method == METHOD_LORA else None,
                "lora_alpha": LORA_ALPHA if method == METHOD_LORA else None,
                "prefix_length": PREFIX_LENGTH if method == METHOD_PREFIX else None,
            },
        },
        output / f"{method}_seed_{seed}.pt",
    )
    return summary
