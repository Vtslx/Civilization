from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from experiments.civilization_transformer_torch.analysis.dataset import LOGIC_LABELS
from experiments.civilization_transformer_torch.model import CivilizationAblationConfig

from ..adapter import CivilizationAdapter, CivilizationAdapterConfig, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import AdapterDiagnosticHeads, LABEL_TO_ID, _kl_preservation
from .answer_option_readout import build_answer_option_vectors
from .evidence_answer_training import _answer_scores, _capture_updates, _margin_loss
from .hidden_states import last_non_padding_pool
from .necessity_alignment_data import NecessityPair


@dataclass(frozen=True)
class MemoryRuleNecessityLossWeights:
    answer_option_margin: float = 1.00
    grounded_context_alignment: float = 0.50
    wrong_context_separation: float = 0.50
    evidence_to_answer_consistency: float = 0.75
    memory_necessity: float = 1.50
    rule_necessity: float = 1.50
    memory_rule_conflict: float = 1.00
    fixed_centroid_group_flip: float = 1.00
    full_hidden_prototype_alignment: float = 1.00
    local_rehearsal_retention: float = 0.75
    logit_preservation: float = 0.10
    residual_budget: float = 0.10


@dataclass
class MemoryRuleNecessityTrainingResult:
    route: str
    seed: int
    target_layers: tuple[int, ...]
    steps: int
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
    checkpoint_path: str
    max_mps_allocated: int


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


def _masked_context_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if vectors.shape[1] == 0:
        return vectors.new_zeros((vectors.shape[0], vectors.shape[-1]))
    weights = mask.to(vectors.dtype).unsqueeze(-1)
    return (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _deterministic_pair_sequence(
    pairs: list[NecessityPair],
    count: int,
    seed: int,
) -> list[NecessityPair]:
    if not pairs:
        raise ValueError("necessity pairs cannot be empty")
    rng = random.Random(seed)
    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair.base_record.source_type,
            pair.pair_type,
            pair.pair_id,
            pair.expected_full_option_id,
        ),
    )
    result: list[NecessityPair] = []
    while len(result) < count:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:count]


def _prototype_losses(
    heads: AdapterDiagnosticHeads,
    pooled: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = F.normalize(heads.logic_prototypes.float(), dim=-1)
    normalized = F.normalize(pooled.float(), dim=-1)
    logits = normalized @ prototypes.transpose(0, 1)
    fixed_centroid_group_flip_loss = F.cross_entropy(logits / 0.1, labels)
    target = prototypes[labels]
    full_hidden_prototype_alignment_loss = (1.0 - (normalized * target).sum(dim=-1)).mean()
    return fixed_centroid_group_flip_loss, full_hidden_prototype_alignment_loss


def _path_drop_loss(
    full_scores: torch.Tensor,
    ablated_scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    full_target = full_scores.gather(1, targets[:, None]).squeeze(1)
    ablated_target = ablated_scores.gather(1, targets[:, None]).squeeze(1)
    return torch.relu(margin + ablated_target - full_target).mean()


def run_memory_rule_necessity_training(
    backend: Qwen3Backend,
    train_pairs: list[NecessityPair],
    output_dir: str | Path,
    route: str,
    seed: int = 202,
    target_layers: tuple[int, ...] = (16, 24),
    steps: int = 120,
    gradient_accumulation: int = 4,
    local_max_length: int = 64,
    external_max_length: int = 384,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    answer_margin: float = 0.20,
    wrong_separation_margin: float = 0.15,
    weights: MemoryRuleNecessityLossWeights | None = None,
) -> tuple[Qwen3MultiAdapterModel, AdapterDiagnosticHeads, MemoryRuleNecessityTrainingResult]:
    if route not in {"local_only_transfer", "external_few_shot"}:
        raise ValueError(f"unsupported route: {route}")
    if steps <= 0 or gradient_accumulation <= 0:
        raise ValueError("steps and gradient_accumulation must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
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
    loss_weights = weights or MemoryRuleNecessityLossWeights()
    sequence = _deterministic_pair_sequence(train_pairs, steps * gradient_accumulation, seed)
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float]] = []
    max_mps_allocated = 0
    cursor = 0

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        for _ in range(gradient_accumulation):
            pair = sequence[cursor]
            cursor += 1
            record = pair.base_record
            samples = [pair.full_sample, pair.counterfactual_sample]
            max_length = external_max_length if record.source_type == "external_benchmark" else local_max_length
            encoded, truncations = backend.encode([qwen_text_for_sample(sample) for sample in samples], max_length=max_length)
            if truncations and record.source_type != "external_benchmark":
                raise ValueError("local necessity training sample was truncated")
            encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                baseline = backend.inference_forward(encoded)
            full_context = context_encoder.build_context(samples, encoded["attention_mask"], context_mode="full")
            no_memory_context = replace(
                full_context,
                ablation_config=CivilizationAblationConfig(use_memory_path=False),
            )
            no_rule_context = replace(
                full_context,
                ablation_config=CivilizationAblationConfig(use_rule_path=False),
            )

            output = model(encoded, full_context)
            no_memory_output = model(encoded, no_memory_context)
            no_rule_output = model(encoded, no_rule_context)
            pooled = last_non_padding_pool(output.hidden_states[-1], output.attention_mask).float()
            baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
            deltas = pooled - baseline_pooled
            updates = _capture_updates(model, output.attention_mask)
            no_memory_delta = (
                last_non_padding_pool(no_memory_output.hidden_states[-1], no_memory_output.attention_mask).float()
                - baseline_pooled
            )
            no_rule_delta = (
                last_non_padding_pool(no_rule_output.hidden_states[-1], no_rule_output.attention_mask).float()
                - baseline_pooled
            )

            if record.answer_options not in option_cache:
                option_cache[record.answer_options] = torch.tensor(
                    build_answer_option_vectors(backend, record.answer_options),
                    dtype=torch.float32,
                    device=backend.device,
                )
            option_vectors = option_cache[record.answer_options]
            targets = torch.tensor(
                [pair.expected_full_option_id, pair.expected_counterfactual_option_id],
                dtype=torch.long,
                device=backend.device,
            )
            scores = _answer_scores(deltas, option_vectors)
            update_scores = _answer_scores(updates[-1], option_vectors)
            no_memory_scores = _answer_scores(no_memory_delta, option_vectors)
            no_rule_scores = _answer_scores(no_rule_delta, option_vectors)
            margin_losses = []
            margins = []
            for index, target in enumerate(targets.tolist()):
                loss, margin = _margin_loss(scores[index : index + 1], target, answer_margin)
                update_loss, update_margin = _margin_loss(update_scores[index : index + 1], target, answer_margin)
                margin_losses.append(0.5 * (loss + update_loss))
                margins.append(0.5 * (margin + update_margin))
            answer_option_margin_loss = torch.stack(margin_losses).mean()
            correct_option_margin = torch.stack(margins).mean()

            evidence_vector = (
                _masked_context_mean(full_context.memory_vectors.float(), full_context.memory_mask)
                + _masked_context_mean(full_context.rule_vectors.float(), full_context.rule_mask)
            )
            grounded_context_alignment_loss = (
                1.0
                - (
                    F.normalize(updates[-1], dim=-1)
                    * F.normalize(evidence_vector, dim=-1)
                ).sum(dim=-1)
            ).mean()

            pair_similarity = (
                F.normalize(deltas[0:1], dim=-1)
                * F.normalize(deltas[1:2], dim=-1)
            ).sum(dim=-1)
            wrong_context_separation_loss = torch.relu(
                pair_similarity - (1.0 - wrong_separation_margin)
            ).mean()
            evidence_to_answer_consistency_loss = 0.5 * (
                _path_drop_loss(scores, no_memory_scores, targets, answer_margin)
                + _path_drop_loss(scores, no_rule_scores, targets, answer_margin)
            )
            memory_necessity_loss = (
                answer_option_margin_loss + _path_drop_loss(scores, no_memory_scores, targets, answer_margin)
                if pair.pair_type == "memory_necessity_pair"
                else pooled.new_zeros(())
            )
            rule_necessity_loss = (
                answer_option_margin_loss + _path_drop_loss(scores, no_rule_scores, targets, answer_margin)
                if pair.pair_type == "rule_necessity_pair"
                else pooled.new_zeros(())
            )
            memory_rule_conflict_loss = (
                answer_option_margin_loss + _path_drop_loss(scores, no_rule_scores, targets, answer_margin)
                if pair.pair_type == "memory_rule_conflict_pair"
                else pooled.new_zeros(())
            )

            labels = torch.tensor(
                [LABEL_TO_ID[sample.label] for sample in samples],
                dtype=torch.long,
                device=backend.device,
            )
            fixed_centroid_group_flip_loss, full_hidden_prototype_alignment_loss = _prototype_losses(
                heads,
                pooled,
                labels,
            )
            local_rehearsal_retention_loss = (
                0.5 * (answer_option_margin_loss + fixed_centroid_group_flip_loss)
                if route == "external_few_shot" and record.source_type == "local_semireal"
                else pooled.new_zeros(())
            )
            preservation = _kl_preservation(output.logits[:, -1, :], baseline.logits[:, -1, :])
            residual_budget = torch.stack(
                [
                    torch.tensor(trace.delta_norm, device=backend.device, dtype=torch.float32)
                    for trace in output.traces.values()
                ]
            ).sum()
            residual_budget_loss = torch.relu(residual_budget - 0.75).square()
            hidden_norm_ratio = max(trace.hidden_norm_ratio for trace in output.traces.values())

            total = (
                loss_weights.answer_option_margin * answer_option_margin_loss
                + loss_weights.grounded_context_alignment * grounded_context_alignment_loss
                + loss_weights.wrong_context_separation * wrong_context_separation_loss
                + loss_weights.evidence_to_answer_consistency * evidence_to_answer_consistency_loss
                + loss_weights.memory_necessity * memory_necessity_loss
                + loss_weights.rule_necessity * rule_necessity_loss
                + loss_weights.memory_rule_conflict * memory_rule_conflict_loss
                + loss_weights.fixed_centroid_group_flip * fixed_centroid_group_flip_loss
                + loss_weights.full_hidden_prototype_alignment * full_hidden_prototype_alignment_loss
                + loss_weights.local_rehearsal_retention * local_rehearsal_retention_loss
                + loss_weights.logit_preservation * preservation
                + loss_weights.residual_budget * residual_budget_loss
            )
            (total / gradient_accumulation).backward()
            values = {
                "total_loss": total,
                "answer_option_margin_loss": answer_option_margin_loss,
                "grounded_context_alignment_loss": grounded_context_alignment_loss,
                "wrong_context_separation_loss": wrong_context_separation_loss,
                "evidence_to_answer_consistency_loss": evidence_to_answer_consistency_loss,
                "memory_necessity_loss": memory_necessity_loss,
                "rule_necessity_loss": rule_necessity_loss,
                "memory_rule_conflict_loss": memory_rule_conflict_loss,
                "fixed_centroid_group_flip_loss": fixed_centroid_group_flip_loss,
                "full_hidden_prototype_alignment_loss": full_hidden_prototype_alignment_loss,
                "local_rehearsal_retention_loss": local_rehearsal_retention_loss,
                "correct_option_margin": correct_option_margin,
                "logit_preservation_kl": preservation,
                "residual_budget_loss": residual_budget_loss,
            }
            for name, value in values.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu()) / gradient_accumulation
            aggregate["hidden_norm_ratio"] = max(aggregate.get("hidden_norm_ratio", 0.0), hidden_norm_ratio)
            for pair_type in ("memory_necessity_pair", "rule_necessity_pair", "memory_rule_conflict_pair"):
                key = f"{pair_type}_count"
                aggregate[key] = aggregate.get(key, 0.0) + (1.0 / gradient_accumulation if pair.pair_type == pair_type else 0.0)

        adapter_gradient_norm = torch.nn.utils.clip_grad_norm_(adapter_parameters + residual_parameters, 1.0)
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
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"memory_rule_necessity_training route={route} step={step + 1}/{steps} "
                f"total_loss={row['total_loss']:.6f} "
                f"answer_margin={row['correct_option_margin']:.6f}",
                flush=True,
            )
        if backend.device.type == "mps":
            torch.mps.empty_cache()

    final_fingerprint = backend.parameter_fingerprint()
    qwen_gradients = sum(parameter.grad is not None for parameter in backend.model.parameters())
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"memory_rule_necessity_{route}_seed_{seed}.pt"
    metadata = {
        "route": route,
        "seed": seed,
        "target_layers": list(model.target_layers),
        "steps": steps,
        "base_model_sha256": backend.initial_sha256,
        "loss_weights": asdict(loss_weights),
    }
    torch.save(_checkpoint_payload(model, heads, metadata), checkpoint_path)
    window = max(1, min(10, len(losses) // 3))
    tracked = (
        "total_loss",
        "memory_necessity_loss",
        "rule_necessity_loss",
        "fixed_centroid_group_flip_loss",
        "answer_option_margin_loss",
    )
    result = MemoryRuleNecessityTrainingResult(
        route=route,
        seed=seed,
        target_layers=model.target_layers,
        steps=steps,
        losses=losses,
        loss_decreased={
            name: _window_mean(losses[-window:], name) < _window_mean(losses[:window], name)
            for name in tracked
        },
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
        checkpoint_path=str(checkpoint_path),
        max_mps_allocated=max_mps_allocated,
    )
    return model, heads, result
