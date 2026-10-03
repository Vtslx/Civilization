from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import psutil
import torch
import torch.nn.functional as F

from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS, LogicSample

from ..adapter import CivilizationAdapter, CivilizationAdapterConfig, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import AdapterDiagnosticHeads, LABEL_TO_ID, _kl_preservation, _labels
from .hidden_states import last_non_padding_pool


DEFAULT_SCENARIO_WEIGHTS = {
    "surface_invariant_label_flip": 0.40,
    "rule_required_priority": 0.25,
    "memory_required_two_hop": 0.0875,
    "state_required_disambiguation": 0.0875,
    "memory_conflict_resolution": 0.0875,
    "counterfactual_memory_swap": 0.0875,
}


@dataclass(frozen=True)
class SurfaceGroupBatch:
    surface_group_id: str
    scenario: str
    stress_profile: str
    samples: tuple[LogicSample, ...]

    def __post_init__(self) -> None:
        if len(self.samples) != len(LOGIC_LABELS):
            raise ValueError("surface group must contain exactly five samples")
        if {sample.label for sample in self.samples} != set(LOGIC_LABELS):
            raise ValueError("surface group must contain each logic label exactly once")
        if any(sample.surface_group_id != self.surface_group_id for sample in self.samples):
            raise ValueError("surface group id mismatch")
        if any(sample.variant != self.scenario for sample in self.samples):
            raise ValueError("surface group scenario mismatch")
        if any(sample.stress_profile != self.stress_profile for sample in self.samples):
            raise ValueError("surface group stress profile mismatch")
        if len({sample.text for sample in self.samples}) != 1:
            raise ValueError("surface group samples must have identical text")
        contexts = {
            (sample.memory_target, sample.state_target, sample.rule_target)
            for sample in self.samples
        }
        if len(contexts) <= 1:
            raise ValueError("surface group contexts must differ")


def build_surface_group_batches(samples: Iterable[LogicSample]) -> tuple[SurfaceGroupBatch, ...]:
    grouped: dict[str, list[LogicSample]] = {}
    for sample in samples:
        if not sample.surface_group_id:
            raise ValueError("surface group id cannot be empty")
        grouped.setdefault(sample.surface_group_id, []).append(sample)
    batches = []
    label_order = {label: index for index, label in enumerate(LOGIC_LABELS)}
    for group_id, rows in sorted(grouped.items()):
        ordered = tuple(sorted(rows, key=lambda sample: label_order[sample.label]))
        if not ordered:
            continue
        batches.append(
            SurfaceGroupBatch(
                surface_group_id=group_id,
                scenario=ordered[0].variant,
                stress_profile=ordered[0].stress_profile,
                samples=ordered,
            )
        )
    return tuple(batches)


class SurfaceGroupSampler:
    def __init__(
        self,
        groups: Iterable[SurfaceGroupBatch],
        seed: int,
        scenario_weights: dict[str, float] | None = None,
    ):
        self.groups = tuple(groups)
        if not self.groups:
            raise ValueError("surface group sampler requires at least one group")
        self.seed = seed
        self.scenario_weights = dict(scenario_weights or DEFAULT_SCENARIO_WEIGHTS)
        by_scenario: dict[str, list[SurfaceGroupBatch]] = {}
        for group in self.groups:
            by_scenario.setdefault(group.scenario, []).append(group)
        missing = set(by_scenario) - set(self.scenario_weights)
        if missing:
            raise ValueError(f"missing scenario weights: {sorted(missing)}")
        active_total = sum(self.scenario_weights[scenario] for scenario in by_scenario)
        if active_total <= 0:
            raise ValueError("active scenario weights must sum to a positive value")
        self._scenarios = tuple(sorted(by_scenario))
        self._weights = tuple(self.scenario_weights[scenario] / active_total for scenario in self._scenarios)
        self._groups_by_scenario = {
            scenario: tuple(sorted(rows, key=lambda group: group.surface_group_id))
            for scenario, rows in by_scenario.items()
        }

    def sequence(self, count: int) -> tuple[SurfaceGroupBatch, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(self.seed)
        scenario_positions = {scenario: 0 for scenario in self._scenarios}
        scenario_groups = {
            scenario: list(groups)
            for scenario, groups in self._groups_by_scenario.items()
        }
        for groups in scenario_groups.values():
            rng.shuffle(groups)
        result = []
        for _ in range(count):
            scenario = rng.choices(self._scenarios, weights=self._weights, k=1)[0]
            rows = scenario_groups[scenario]
            position = scenario_positions[scenario]
            if position and position % len(rows) == 0:
                rng.shuffle(rows)
            result.append(rows[position % len(rows)])
            scenario_positions[scenario] = position + 1
        return tuple(result)


@dataclass(frozen=True)
class SurfaceGroupLossWeights:
    group_target_alignment: float = 1.0
    group_context_separation: float = 1.0
    group_all_correct: float = 0.75
    context_delta_direction: float = 0.75
    cross_layer_flip_consistency: float = 0.50
    group_collapse_penalty: float = 0.50


@dataclass
class SurfaceGroupLosses:
    group_target_alignment_loss: torch.Tensor
    group_context_separation_loss: torch.Tensor
    group_all_correct_loss: torch.Tensor
    context_delta_direction_loss: torch.Tensor
    cross_layer_flip_consistency_loss: torch.Tensor
    group_collapse_penalty: torch.Tensor
    prototype_margin: torch.Tensor
    group_min_representation_distance: torch.Tensor

    def weighted_total(self, weights: SurfaceGroupLossWeights) -> torch.Tensor:
        return (
            weights.group_target_alignment * self.group_target_alignment_loss
            + weights.group_context_separation * self.group_context_separation_loss
            + weights.group_all_correct * self.group_all_correct_loss
            + weights.context_delta_direction * self.context_delta_direction_loss
            + weights.cross_layer_flip_consistency * self.cross_layer_flip_consistency_loss
            + weights.group_collapse_penalty * self.group_collapse_penalty
        )


def compute_surface_group_losses(
    layer_updates: list[torch.Tensor],
    final_deltas: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    separation_margin: float = 0.25,
) -> SurfaceGroupLosses:
    if not layer_updates:
        raise ValueError("layer_updates cannot be empty")
    if final_deltas.ndim != 2:
        raise ValueError("final_deltas must have shape [batch, hidden]")
    if final_deltas.shape[0] != len(LOGIC_LABELS):
        raise ValueError("surface group loss requires five samples")
    if labels.shape != (len(LOGIC_LABELS),):
        raise ValueError("labels must have shape [5]")
    normalized_prototypes = F.normalize(prototypes.float(), dim=-1)
    target_prototypes = normalized_prototypes[labels]
    normalized_updates = [F.normalize(update.float(), dim=-1) for update in layer_updates]
    normalized_delta = F.normalize(final_deltas.float(), dim=-1)

    target_alignment = torch.stack(
        [(1.0 - (update * target_prototypes).sum(dim=-1)).mean() for update in normalized_updates]
    ).mean()
    context_delta_direction = (1.0 - (normalized_delta * target_prototypes).sum(dim=-1)).mean()
    prototype_logits = normalized_delta @ normalized_prototypes.transpose(0, 1)
    group_all_correct = F.cross_entropy(prototype_logits / 0.1, labels)

    pair_mask = ~torch.eye(final_deltas.shape[0], dtype=torch.bool, device=final_deltas.device)
    similarities = normalized_delta @ normalized_delta.transpose(0, 1)
    pair_distances = (1.0 - similarities)[pair_mask]
    context_separation = torch.relu(separation_margin - pair_distances).mean()
    collapse_penalty = torch.relu(0.10 - pair_distances).mean()

    cross_layer = final_deltas.new_zeros(())
    if len(normalized_updates) > 1:
        pairs = []
        for left_index, left in enumerate(normalized_updates):
            for right in normalized_updates[left_index + 1 :]:
                pairs.append(1.0 - (left * right).sum(dim=-1).mean())
        cross_layer = torch.stack(pairs).mean()

    positive = prototype_logits.gather(1, labels[:, None]).squeeze(1)
    negative_mask = F.one_hot(labels, num_classes=prototypes.shape[0]).bool()
    hardest_negative = prototype_logits.masked_fill(negative_mask, -1e4).max(dim=-1).values
    prototype_margin = (positive - hardest_negative).mean()
    return SurfaceGroupLosses(
        group_target_alignment_loss=target_alignment,
        group_context_separation_loss=context_separation,
        group_all_correct_loss=group_all_correct,
        context_delta_direction_loss=context_delta_direction,
        cross_layer_flip_consistency_loss=cross_layer,
        group_collapse_penalty=collapse_penalty,
        prototype_margin=prototype_margin,
        group_min_representation_distance=pair_distances.min(),
    )


@dataclass
class Qwen3SurfaceFlipTrainingResult:
    seed: int
    target_layers: tuple[int, ...]
    steps: int
    training_mode: str
    losses: list[dict[str, float]]
    loss_decreased: dict[str, bool]
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


def _window_mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return sum(values) / len(values) if values else 0.0


def _checkpoint_payload(
    model: Qwen3MultiAdapterModel,
    heads: AdapterDiagnosticHeads,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "adapter_state_dict": model.adapters.state_dict(),
        "diagnostic_heads_state_dict": heads.state_dict(),
        "adapter_configs": {
            str(layer): model.adapters[str(layer)].config.to_dict()
            for layer in model.target_layers
        },
        "training_metadata": metadata,
    }


def run_qwen3_surface_flip_training(
    backend: Qwen3Backend,
    train_samples: list[LogicSample],
    output_dir: str | Path,
    target_layers: tuple[int, ...] = (16, 24),
    seed: int = 202,
    steps: int = 140,
    group_batch_size: int = 1,
    gradient_accumulation: int = 4,
    max_length: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    separation_margin: float = 0.25,
    loss_weights: SurfaceGroupLossWeights | None = None,
    scenario_weights: dict[str, float] | None = None,
    training_mode: str = "surface_group_flip_alignment",
) -> tuple[Qwen3MultiAdapterModel, AdapterDiagnosticHeads, Qwen3SurfaceFlipTrainingResult]:
    if training_mode not in {"surface_group_flip_alignment", "stage25_baseline"}:
        raise ValueError(f"unsupported training mode: {training_mode}")
    if group_batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("group_batch_size and gradient_accumulation must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    groups = build_surface_group_batches(train_samples)
    sampler = SurfaceGroupSampler(groups, seed=seed, scenario_weights=scenario_weights)
    sampled_groups = sampler.sequence(steps * gradient_accumulation * group_batch_size)
    adapters = {
        layer: CivilizationAdapter(CivilizationAdapterConfig(target_layer=layer))
        for layer in target_layers
    }
    model = Qwen3MultiAdapterModel(backend, adapters)
    heads = AdapterDiagnosticHeads().to(backend.device, dtype=torch.float32)
    context_encoder = FrozenQwenContextEncoder(backend)
    residual_parameters = [adapter.residual_scale for adapter in model.adapters.values()]
    adapter_parameters = [
        parameter
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
        if parameter is not adapter.residual_scale
    ]
    head_parameters = list(heads.parameters())
    trainable = adapter_parameters + residual_parameters + head_parameters
    qwen_ids = {id(parameter) for parameter in backend.model.parameters()}
    optimizer_contains_qwen = any(id(parameter) in qwen_ids for parameter in trainable)
    if optimizer_contains_qwen:
        raise RuntimeError("optimizer contains frozen Qwen parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": residual_parameters, "lr": learning_rate * 10.0, "weight_decay": 0.0},
            {"params": head_parameters, "lr": learning_rate, "weight_decay": weight_decay},
        ]
    )
    weights = loss_weights or SurfaceGroupLossWeights()
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float]] = []
    max_mps_allocated = 0
    cursor = 0

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        for _micro_step in range(gradient_accumulation):
            micro_groups = sampled_groups[cursor : cursor + group_batch_size]
            cursor += group_batch_size
            samples = [sample for group in micro_groups for sample in group.samples]
            encoded, truncations = backend.encode(
                [qwen_text_for_sample(sample) for sample in samples],
                max_length=max_length,
            )
            if truncations:
                raise ValueError("surface group training sample was truncated")
            encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                baseline = backend.inference_forward(encoded)
            context = context_encoder.build_context(samples, encoded["attention_mask"])
            output = backend.adapter_training_forward(encoded, model, context)
            pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
            baseline_pooled = last_non_padding_pool(
                baseline.hidden_states[-1],
                baseline.attention_mask,
            ).float()
            labels = _labels(samples, backend.device)
            classification = F.cross_entropy(heads.classifier(pooled), labels)
            normalized_prototypes = F.normalize(heads.logic_prototypes, dim=-1)
            target_prototype = normalized_prototypes[labels]
            layer_updates = []
            residual_budget = pooled.new_zeros(())
            for layer in model.target_layers:
                adapter = model.adapters[str(layer)]
                if adapter.last_pre_scale_update is None:
                    raise RuntimeError(f"adapter {layer} did not expose its pre-scale update")
                layer_updates.append(
                    last_non_padding_pool(adapter.last_pre_scale_update, output.attention_mask).float()
                )
                residual_budget = residual_budget + torch.tensor(
                    output.traces[layer].delta_norm,
                    dtype=pooled.dtype,
                    device=backend.device,
                )
            normalized_last_update = F.normalize(layer_updates[-1], dim=-1)
            direction_loss = (1.0 - (normalized_last_update * target_prototype).sum(dim=-1)).mean()
            prototype_similarity = normalized_last_update @ normalized_prototypes.transpose(0, 1)
            positive = prototype_similarity.gather(1, labels[:, None])
            negative_mask = F.one_hot(labels, num_classes=len(LOGIC_LABELS)).bool()
            hardest_negative = prototype_similarity.masked_fill(negative_mask, -1e4).max(dim=-1, keepdim=True).values
            hard_negative = torch.relu(0.2 + hardest_negative - positive).mean()
            final_delta = pooled - baseline_pooled
            final_delta_target = target_prototype * 0.25
            final_delta_loss = F.mse_loss(final_delta, final_delta_target) * pooled.shape[-1]
            centroid_loss = direction_loss + final_delta_loss

            required_paths = {path for sample in samples for path in sample.required_paths}
            memory_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.memory(pooled.detach()), labels)
                if "memory" in required_paths
                else pooled.new_zeros(())
            )
            state_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.state(pooled.detach()), labels)
                if "state" in required_paths
                else pooled.new_zeros(())
            )
            rule_loss = (
                direction_loss + 0.1 * F.cross_entropy(heads.rule(pooled.detach()), labels)
                if "rule" in required_paths
                else pooled.new_zeros(())
            )
            flip_loss = final_delta_loss + 0.1 * F.cross_entropy(heads.flip(pooled.detach()), labels)
            preservation = _kl_preservation(output.logits[:, -1, :], baseline.logits[:, -1, :])
            hidden_ratio = torch.stack(
                [
                    torch.tensor(trace.hidden_norm_ratio, dtype=pooled.dtype, device=backend.device)
                    for trace in output.traces.values()
                ]
            ).max()
            norm_penalty = torch.relu(hidden_ratio - 1.25).square()
            residual_budget_loss = torch.relu(residual_budget - 0.75).square()

            old_cross_layer = pooled.new_zeros(())
            if len(layer_updates) > 1:
                normalized_layers = [F.normalize(update, dim=-1) for update in layer_updates]
                old_cross_layer = torch.stack(
                    [
                        1.0 - (left * right).sum(dim=-1).mean()
                        for left_index, left in enumerate(normalized_layers)
                        for right in normalized_layers[left_index + 1 :]
                    ]
                ).mean()
            stage25_total = (
                0.2 * classification
                + 0.7 * memory_loss
                + 0.7 * state_loss
                + 0.7 * rule_loss
                + 0.8 * flip_loss
                + centroid_loss
                + 0.5 * hard_negative
                + 0.5 * old_cross_layer
                + 0.1 * preservation
                + 0.1 * norm_penalty
                + 0.1 * residual_budget_loss
            )

            group_loss_values = []
            group_metric_values = []
            for group_index in range(group_batch_size):
                start = group_index * len(LOGIC_LABELS)
                end = start + len(LOGIC_LABELS)
                group_losses = compute_surface_group_losses(
                    [update[start:end] for update in layer_updates],
                    final_delta[start:end],
                    labels[start:end],
                    heads.logic_prototypes,
                    separation_margin=separation_margin,
                )
                group_loss_values.append(group_losses.weighted_total(weights))
                group_metric_values.append(group_losses)
            group_total = torch.stack(group_loss_values).mean()
            if training_mode == "stage25_baseline":
                group_total = group_total * 0.0
            total = stage25_total + group_total
            (total / gradient_accumulation).backward()

            group_metrics = {
                name: torch.stack([getattr(item, name) for item in group_metric_values]).mean()
                for name in (
                    "group_target_alignment_loss",
                    "group_context_separation_loss",
                    "group_all_correct_loss",
                    "context_delta_direction_loss",
                    "cross_layer_flip_consistency_loss",
                    "group_collapse_penalty",
                    "prototype_margin",
                    "group_min_representation_distance",
                )
            }
            values = {
                "classification_loss": classification,
                "memory_dependency_loss": memory_loss,
                "state_dependency_loss": state_loss,
                "rule_dependency_loss": rule_loss,
                "context_flip_loss": flip_loss,
                "centroid_separation_loss": centroid_loss,
                "hard_negative_loss": hard_negative,
                "cross_layer_consistency_loss": old_cross_layer,
                "logit_preservation_kl": preservation,
                "adapter_norm_penalty": norm_penalty,
                "residual_budget_loss": residual_budget_loss,
                "stage25_existing_loss": stage25_total,
                **group_metrics,
                "surface_group_loss": group_total,
                "total_loss": total,
                "hidden_norm_ratio": hidden_ratio,
            }
            for name, value in values.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu()) / gradient_accumulation
            del baseline, output, pooled, baseline_pooled, context

        adapter_gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter_parameters + residual_parameters,
            1.0,
        )
        head_gradient_norm = torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
        optimizer.step()
        mps_allocated = torch.mps.current_allocated_memory() if backend.device.type == "mps" else 0
        row = {
            "step": float(step),
            **aggregate,
            "adapter_gradient_norm": float(adapter_gradient_norm.detach().cpu()),
            "head_gradient_norm": float(head_gradient_norm.detach().cpu()),
            "rss": float(psutil.Process().memory_info().rss),
            "mps_allocated": float(mps_allocated),
        }
        for layer in model.target_layers:
            row[f"residual_scale_{layer}"] = float(model.adapters[str(layer)].residual_scale.detach().cpu())
        losses.append(row)
        max_mps_allocated = max(max_mps_allocated, int(mps_allocated))
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"surface_flip_dual_16_24_seed_{seed}.pt"
    metadata = {
        "seed": seed,
        "target_layers": list(model.target_layers),
        "steps": steps,
        "training_mode": training_mode,
        "base_model_sha256": backend.initial_sha256,
        "loss_weights": asdict(weights),
    }
    torch.save(_checkpoint_payload(model, heads, metadata), checkpoint_path)
    window = max(1, min(10, len(losses) // 3))
    tracked_losses = (
        "total_loss",
        "classification_loss",
        "context_flip_loss",
        "group_target_alignment_loss",
        "group_context_separation_loss",
        "group_all_correct_loss",
        "context_delta_direction_loss",
        "cross_layer_flip_consistency_loss",
        "group_collapse_penalty",
    )
    loss_decreased = {
        name: _window_mean(losses[-window:], name) < _window_mean(losses[:window], name)
        for name in tracked_losses
    }
    result = Qwen3SurfaceFlipTrainingResult(
        seed=seed,
        target_layers=model.target_layers,
        steps=steps,
        training_mode=training_mode,
        losses=losses,
        loss_decreased=loss_decreased,
        adapter_parameter_count=sum(adapter.trainable_parameter_count for adapter in model.adapters.values()),
        qwen_parameter_count=sum(parameter.numel() for parameter in backend.model.parameters()),
        qwen_trainable_parameter_count=sum(
            parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad
        ),
        qwen_gradients_present=qwen_gradients,
        optimizer_contains_qwen_parameters=optimizer_contains_qwen,
        initial_fingerprint=initial_fingerprint,
        final_fingerprint=final_fingerprint,
        fingerprint_unchanged=initial_fingerprint == final_fingerprint,
        max_mps_allocated=max_mps_allocated,
        checkpoint_path=str(checkpoint_path),
    )
    return model, heads, result


def surface_flip_checkpoint_contains_qwen_weights(path: str | Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    forbidden = ("model.layers.", "model.embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj")
    keys = []
    for section in ("adapter_state_dict", "diagnostic_heads_state_dict"):
        keys.extend(payload.get(section, {}).keys())
    return any(any(token in key for token in forbidden) for key in keys)
