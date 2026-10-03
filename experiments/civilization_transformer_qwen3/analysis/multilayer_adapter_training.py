from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from ..adapter import CivilizationAdapter, CivilizationAdapterConfig, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import (
    AdapterDiagnosticHeads,
    LABEL_TO_ID,
    _kl_preservation,
    _labels,
    _next_sample,
)
from .hidden_states import last_non_padding_pool


@dataclass
class Qwen3MultiAdapterTrainingResult:
    seed: int
    config_name: str
    target_layers: tuple[int, ...]
    steps: int
    losses: list[dict[str, float]]
    initial_total_loss: float
    final_total_loss: float
    total_loss_decreased: bool
    initial_classification_loss: float
    final_classification_loss: float
    classification_loss_decreased: bool
    initial_cross_layer_consistency_loss: float
    final_cross_layer_consistency_loss: float
    cross_layer_consistency_decreased: bool
    adapter_parameter_count: int
    qwen_parameter_count: int
    qwen_trainable_parameter_count: int
    qwen_gradients_present: int
    optimizer_contains_qwen_parameters: bool
    initial_fingerprint: dict[str, float | int]
    final_fingerprint: dict[str, float | int]
    fingerprint_unchanged: bool
    max_mps_allocated: int
    checkpoint_path: str


def _checkpoint_payload(
    model: Qwen3MultiAdapterModel,
    heads: AdapterDiagnosticHeads,
    result_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "adapter_state_dict": model.adapters.state_dict(),
        "diagnostic_heads_state_dict": heads.state_dict(),
        "adapter_configs": {
            str(layer): model.adapters[str(layer)].config.to_dict()
            for layer in model.target_layers
        },
        "training_metadata": result_meta,
    }


def _window_mean(key: str, rows: list[dict[str, float]]) -> float:
    values = [row[key] for row in rows if row[key] > 0]
    return sum(values) / len(values) if values else 0.0


def run_qwen3_multilayer_adapter_training(
    backend: Qwen3Backend,
    train_samples,
    config_name: str,
    target_layers: tuple[int, ...],
    output_dir: str | Path,
    seed: int,
    steps: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    gradient_accumulation: int = 8,
    max_length: int = 64,
) -> tuple[Qwen3MultiAdapterModel, AdapterDiagnosticHeads, Qwen3MultiAdapterTrainingResult]:
    if not train_samples:
        raise ValueError("train_samples cannot be empty")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("target_layers must be unique")
    torch.manual_seed(seed)
    random.seed(seed)
    adapters = {
        layer: CivilizationAdapter(CivilizationAdapterConfig(target_layer=layer))
        for layer in target_layers
    }
    single_counts = {layer: adapter.trainable_parameter_count for layer, adapter in adapters.items()}
    if any(count >= 2_000_000 for count in single_counts.values()):
        raise ValueError("single CivilizationAdapter exceeds the 2M parameter limit")
    if sum(single_counts.values()) >= 4_000_000 and len(adapters) == 2:
        raise ValueError("dual CivilizationAdapter exceeds the 4M parameter limit")
    model = Qwen3MultiAdapterModel(backend, adapters)
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    context_encoder = FrozenQwenContextEncoder(backend)
    residual_scale_parameters = [
        adapter.residual_scale for adapter in model.adapters.values()
    ]
    adapter_parameters = [
        parameter
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
        if parameter is not adapter.residual_scale
    ]
    head_parameters = list(heads.parameters())
    trainable = adapter_parameters + residual_scale_parameters + head_parameters
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    optimizer_contains_qwen = any(id(parameter) in qwen_ids for parameter in trainable)
    if optimizer_contains_qwen:
        raise RuntimeError("optimizer contains frozen Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": residual_scale_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": head_parameters, "lr": learning_rate, "weight_decay": weight_decay},
        ]
    )
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float]] = []
    max_mps_allocated = 0
    metric_names = (
        "classification_loss",
        "memory_dependency_loss",
        "state_dependency_loss",
        "rule_dependency_loss",
        "context_flip_loss",
        "centroid_separation_loss",
        "hard_negative_loss",
        "cross_layer_consistency_loss",
        "logit_preservation_kl",
        "adapter_norm_penalty",
        "residual_budget_loss",
        "total_loss",
        "hidden_norm_ratio",
    )

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        aggregates = {name: 0.0 for name in metric_names}
        for micro_step in range(gradient_accumulation):
            sample_index = step * gradient_accumulation + micro_step
            sample = _next_sample(train_samples, sample_index, seed)
            encoded, truncations = backend.encode([qwen_text_for_sample(sample)], max_length=max_length)
            if truncations:
                raise ValueError("training sample was truncated")
            encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                baseline = backend.inference_forward(encoded)
            context = context_encoder.build_context([sample], encoded["attention_mask"])
            output = backend.adapter_training_forward(encoded, model, context)
            pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
            labels = _labels([sample], backend.device)
            classification = F.cross_entropy(heads.classifier(pooled), labels)
            zero = torch.zeros((), device=backend.device, dtype=pooled.dtype)

            normalized_prototypes = F.normalize(heads.logic_prototypes, dim=-1)
            target_prototype = normalized_prototypes[labels]
            per_layer_directions = []
            per_layer_direction_losses = []
            residual_budget = zero
            for layer in model.target_layers:
                adapter = model.adapters[str(layer)]
                if adapter.last_pre_scale_update is None:
                    raise RuntimeError(f"adapter {layer} did not expose its pre-scale update")
                update_pooled = last_non_padding_pool(
                    adapter.last_pre_scale_update,
                    output.attention_mask,
                ).float()
                normalized_hidden = F.normalize(update_pooled, dim=-1)
                per_layer_directions.append(normalized_hidden)
                per_layer_directions_loss = (1.0 - (normalized_hidden * target_prototype).sum(dim=-1)).mean()
                per_layer_direction_losses.append(per_layer_directions_loss)
                residual_budget = residual_budget + torch.tensor(
                    output.traces[layer].delta_norm,
                    dtype=pooled.dtype,
                    device=backend.device,
                )
            direction_loss = torch.stack(per_layer_direction_losses).mean()
            cross_layer = zero
            if len(per_layer_directions) > 1:
                pairs = []
                for left_index, left in enumerate(per_layer_directions):
                    for right in per_layer_directions[left_index + 1 :]:
                        pairs.append(1.0 - (left * right).sum(dim=-1).mean())
                cross_layer = torch.stack(pairs).mean()

            prototype_similarity = per_layer_directions[-1] @ normalized_prototypes.transpose(0, 1)
            positive_scores = prototype_similarity.gather(1, labels[:, None])
            negative_mask = F.one_hot(labels, num_classes=len(LABEL_TO_ID)).bool()
            hardest_negative = prototype_similarity.masked_fill(negative_mask, -1e4).max(dim=-1, keepdim=True).values
            hard_negative = torch.relu(0.2 + hardest_negative - positive_scores).mean()
            preservation = _kl_preservation(output.logits[:, -1, :], baseline.logits[:, -1, :])
            baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask)
            final_delta = pooled - baseline_pooled.float()
            final_delta_target = target_prototype * 0.25
            final_delta_loss = F.mse_loss(final_delta, final_delta_target) * pooled.shape[-1]
            centroid = direction_loss + final_delta_loss
            memory_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.memory(pooled.detach()), labels)
                if "memory" in sample.required_paths
                else zero
            )
            state_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.state(pooled.detach()), labels)
                if "state" in sample.required_paths
                else zero
            )
            rule_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.rule(pooled.detach()), labels)
                if "rule" in sample.required_paths
                else zero
            )
            flip_loss = (
                final_delta_loss + 0.1 * F.cross_entropy(heads.flip(pooled.detach()), labels)
                if sample.variant == "surface_invariant_label_flip"
                else zero
            )
            hidden_ratio_values = [
                torch.tensor(trace.hidden_norm_ratio, dtype=pooled.dtype, device=backend.device)
                for trace in output.traces.values()
            ]
            hidden_ratio = torch.stack(hidden_ratio_values).max()
            norm_penalty = torch.relu(hidden_ratio - 1.25).square()
            residual_budget_loss = torch.relu(residual_budget - 0.75).square()
            total = (
                0.2 * classification
                + 0.7 * memory_loss
                + 0.7 * state_loss
                + 0.7 * rule_loss
                + 0.8 * flip_loss
                + centroid
                + 0.5 * hard_negative
                + 0.5 * cross_layer
                + 0.1 * preservation
                + 0.1 * norm_penalty
                + 0.1 * residual_budget_loss
            )
            (total / gradient_accumulation).backward()
            values = {
                "classification_loss": classification,
                "memory_dependency_loss": memory_loss,
                "state_dependency_loss": state_loss,
                "rule_dependency_loss": rule_loss,
                "context_flip_loss": flip_loss,
                "centroid_separation_loss": centroid,
                "hard_negative_loss": hard_negative,
                "cross_layer_consistency_loss": cross_layer,
                "logit_preservation_kl": preservation,
                "adapter_norm_penalty": norm_penalty,
                "residual_budget_loss": residual_budget_loss,
                "total_loss": total,
                "hidden_norm_ratio": hidden_ratio,
            }
            for name, value in values.items():
                aggregates[name] += float(value.detach().cpu()) / gradient_accumulation
            del baseline, output, pooled, context

        adapter_gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter_parameters + residual_scale_parameters,
            1.0,
        )
        head_gradient_norm = torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
        optimizer.step()
        mps_allocated = torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0
        row = {
            "step": float(step),
            **aggregates,
            "adapter_gradient_norm": float(adapter_gradient_norm.detach().cpu()),
            "head_gradient_norm": float(head_gradient_norm.detach().cpu()),
            "rss": float(psutil.Process().memory_info().rss),
            "mps_allocated": float(mps_allocated),
        }
        for layer in model.target_layers:
            row[f"residual_scale_{layer}"] = float(
                model.adapters[str(layer)].residual_scale.detach().cpu()
            )
        losses.append(row)
        max_mps_allocated = max(max_mps_allocated, int(mps_allocated))
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"multilayer_{config_name}_seed_{seed}.pt"
    metadata = {
        "seed": seed,
        "config_name": config_name,
        "target_layers": list(model.target_layers),
        "steps": steps,
        "base_model_sha256": backend.initial_sha256,
    }
    torch.save(_checkpoint_payload(model, heads, metadata), checkpoint_path)

    window = max(1, min(10, len(losses) // 3))
    initial_rows = losses[:window]
    final_rows = losses[-window:]
    initial_total = _window_mean("total_loss", initial_rows)
    final_total = _window_mean("total_loss", final_rows)
    initial_classification = _window_mean("classification_loss", initial_rows)
    final_classification = _window_mean("classification_loss", final_rows)
    initial_consistency = _window_mean("cross_layer_consistency_loss", initial_rows)
    final_consistency = _window_mean("cross_layer_consistency_loss", final_rows)
    adapter_parameter_count = sum(
        adapter.trainable_parameter_count for adapter in model.adapters.values()
    )
    result = Qwen3MultiAdapterTrainingResult(
        seed=seed,
        config_name=config_name,
        target_layers=model.target_layers,
        steps=steps,
        losses=losses,
        initial_total_loss=initial_total,
        final_total_loss=final_total,
        total_loss_decreased=final_total < initial_total,
        initial_classification_loss=initial_classification,
        final_classification_loss=final_classification,
        classification_loss_decreased=final_classification < initial_classification,
        initial_cross_layer_consistency_loss=initial_consistency,
        final_cross_layer_consistency_loss=final_consistency,
        cross_layer_consistency_decreased=final_consistency < initial_consistency if len(model.target_layers) > 1 else True,
        adapter_parameter_count=adapter_parameter_count,
        qwen_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters()),
        qwen_trainable_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad),
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=optimizer_contains_qwen,
        initial_fingerprint=initial_fingerprint,
        final_fingerprint=final_fingerprint,
        fingerprint_unchanged=initial_fingerprint == final_fingerprint,
        max_mps_allocated=max_mps_allocated,
        checkpoint_path=str(checkpoint_path),
    )
    return model, heads, result


def multilayer_checkpoint_contains_qwen_weights(path: str | Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    forbidden = ("model.layers.", "model.embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj")
    keys = []
    for section in ("adapter_state_dict", "diagnostic_heads_state_dict"):
        keys.extend(payload.get(section, {}).keys())
    return any(any(token in key for token in forbidden) for key in keys)
