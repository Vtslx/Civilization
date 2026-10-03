from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any

import psutil
import torch
from torch import nn
import torch.nn.functional as F

from experiments.civilization_transformer_torch.model import CivilizationAblationConfig

from ..adapter import CivilizationAdapter, CivilizationAdapterConfig, Qwen3MultiAdapterModel
from ..adapter.context_encoder import FrozenQwenContextEncoder, qwen_text_for_sample
from ..backend import Qwen3Backend
from .adapter_training import AdapterDiagnosticHeads, LABEL_TO_ID, _kl_preservation
from .answer_option_readout import build_answer_option_vectors
from .evidence_answer_data import EvidenceAnswerSample, wrong_context_sample
from .hidden_states import last_non_padding_pool


@dataclass(frozen=True)
class EvidenceAnswerLossWeights:
    existing_stage28: float = 0.50
    answer_option_margin: float = 1.50
    grounded_context_alignment: float = 1.00
    wrong_context_separation: float = 1.00
    wrong_context_answer: float = 1.00
    evidence_to_answer_consistency: float = 1.00
    logit_preservation: float = 0.10
    residual_budget: float = 0.10


@dataclass
class EvidenceAnswerTrainingResult:
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


def _answer_scores(vector: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.float(), dim=-1) @ F.normalize(options.float(), dim=-1).transpose(0, 1)


def _margin_loss(scores: torch.Tensor, target: int, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    positive = scores[:, target]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[:, target] = False
    hardest_negative = scores.masked_fill(~mask, -1e4).max(dim=-1).values
    return torch.relu(margin + hardest_negative - positive).mean(), (positive - hardest_negative).mean()


def _capture_updates(model: Qwen3MultiAdapterModel, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    updates = []
    for layer in model.target_layers:
        update = model.adapters[str(layer)].last_pre_scale_update
        if update is None:
            raise RuntimeError(f"adapter {layer} did not expose its update")
        updates.append(last_non_padding_pool(update, attention_mask).float())
    return updates


def _deterministic_sequence(
    records: list[EvidenceAnswerSample],
    count: int,
    seed: int,
) -> list[EvidenceAnswerSample]:
    if not records:
        raise ValueError("training records cannot be empty")
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda row: (row.source_type, row.sample.variant, row.context_pair_id, row.correct_option_id))
    result = []
    while len(result) < count:
        current = ordered.copy()
        rng.shuffle(current)
        result.extend(current)
    return result[:count]


def run_evidence_answer_alignment_training(
    backend: Qwen3Backend,
    train_records: list[EvidenceAnswerSample],
    output_dir: str | Path,
    route: str,
    seed: int = 202,
    target_layers: tuple[int, ...] = (16, 24),
    steps: int = 100,
    gradient_accumulation: int = 4,
    local_max_length: int = 64,
    external_max_length: int = 384,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    answer_margin: float = 0.20,
    wrong_separation_margin: float = 0.15,
    weights: EvidenceAnswerLossWeights | None = None,
) -> tuple[Qwen3MultiAdapterModel, AdapterDiagnosticHeads, EvidenceAnswerTrainingResult]:
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
    loss_weights = weights or EvidenceAnswerLossWeights()
    sequence = _deterministic_sequence(train_records, steps * gradient_accumulation, seed)
    option_cache: dict[tuple[str, ...], torch.Tensor] = {}
    initial_fingerprint = backend.parameter_fingerprint()
    losses: list[dict[str, float]] = []
    max_mps_allocated = 0
    cursor = 0

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        for _ in range(gradient_accumulation):
            record = sequence[cursor]
            cursor += 1
            sample = record.sample
            max_length = external_max_length if record.source_type == "external_benchmark" else local_max_length
            encoded, truncations = backend.encode([qwen_text_for_sample(sample)], max_length=max_length)
            if truncations and record.source_type != "external_benchmark":
                raise ValueError("local evidence-answer training sample was truncated")
            encoded = {name: tensor.to(backend.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                baseline = backend.inference_forward(encoded)
            full_context = context_encoder.build_context([sample], encoded["attention_mask"], context_mode="full")
            wrong_context = context_encoder.build_context(
                [wrong_context_sample(record)],
                encoded["attention_mask"],
                context_mode="full",
            )
            no_memory_context = replace(
                full_context,
                ablation_config=CivilizationAblationConfig(use_memory_path=False),
            )
            no_rule_context = replace(
                full_context,
                ablation_config=CivilizationAblationConfig(use_rule_path=False),
            )

            full_output = model(encoded, full_context)
            full_pooled = last_non_padding_pool(full_output.hidden_states[-1], full_output.attention_mask).float()
            baseline_pooled = last_non_padding_pool(baseline.hidden_states[-1], baseline.attention_mask).float()
            full_delta = full_pooled - baseline_pooled
            full_updates = _capture_updates(model, full_output.attention_mask)

            wrong_output = model(encoded, wrong_context)
            wrong_pooled = last_non_padding_pool(wrong_output.hidden_states[-1], wrong_output.attention_mask).float()
            wrong_delta = wrong_pooled - baseline_pooled
            wrong_updates = _capture_updates(model, wrong_output.attention_mask)

            no_memory_output = model(encoded, no_memory_context)
            no_memory_delta = (
                last_non_padding_pool(no_memory_output.hidden_states[-1], no_memory_output.attention_mask).float()
                - baseline_pooled
            )
            no_rule_output = model(encoded, no_rule_context)
            no_rule_delta = (
                last_non_padding_pool(no_rule_output.hidden_states[-1], no_rule_output.attention_mask).float()
                - baseline_pooled
            )

            if record.answer_options not in option_cache:
                frozen_options = build_answer_option_vectors(backend, record.answer_options)
                option_cache[record.answer_options] = torch.tensor(
                    frozen_options,
                    dtype=torch.float32,
                    device=backend.device,
                )
            option_vectors = option_cache[record.answer_options]
            target = record.correct_option_id
            final_scores = _answer_scores(full_delta, option_vectors)
            update_scores = _answer_scores(full_updates[-1], option_vectors)
            final_margin_loss, final_margin = _margin_loss(final_scores, target, answer_margin)
            update_margin_loss, update_margin = _margin_loss(update_scores, target, answer_margin)
            answer_option_margin_loss = 0.5 * (final_margin_loss + update_margin_loss)

            evidence_vector = (
                _masked_context_mean(full_context.memory_vectors.float(), full_context.memory_mask)
                + _masked_context_mean(full_context.rule_vectors.float(), full_context.rule_mask)
            )
            grounded_context_alignment_loss = (
                1.0
                - (
                    F.normalize(full_updates[-1], dim=-1)
                    * F.normalize(evidence_vector, dim=-1)
                ).sum(dim=-1)
            ).mean()

            wrong_similarity = (
                F.normalize(full_delta, dim=-1)
                * F.normalize(wrong_delta, dim=-1)
            ).sum(dim=-1)
            wrong_context_separation_loss = torch.relu(
                wrong_similarity - (1.0 - wrong_separation_margin)
            ).mean()
            wrong_scores = _answer_scores(wrong_delta, option_vectors)
            wrong_target_loss, _wrong_margin = _margin_loss(
                wrong_scores,
                record.wrong_context_option_id,
                answer_margin,
            )
            wrong_context_answer_loss = wrong_target_loss + torch.relu(
                answer_margin + wrong_scores[:, target] - final_scores[:, target]
            ).mean()

            no_memory_scores = _answer_scores(no_memory_delta, option_vectors)
            no_rule_scores = _answer_scores(no_rule_delta, option_vectors)
            evidence_to_answer_consistency_loss = 0.5 * (
                torch.relu(answer_margin + no_memory_scores[:, target] - final_scores[:, target]).mean()
                + torch.relu(answer_margin + no_rule_scores[:, target] - final_scores[:, target]).mean()
            )

            label = torch.tensor([LABEL_TO_ID[sample.label]], dtype=torch.long, device=backend.device)
            classification = F.cross_entropy(heads.classifier(full_pooled), label)
            logic_prototype = F.normalize(heads.logic_prototypes[label], dim=-1)
            logic_direction = (
                1.0
                - (F.normalize(full_updates[-1], dim=-1) * logic_prototype).sum(dim=-1)
            ).mean()
            cross_layer = (
                1.0
                - (
                    F.normalize(full_updates[0], dim=-1)
                    * F.normalize(full_updates[-1], dim=-1)
                ).sum(dim=-1)
            ).mean()
            context_flip = torch.relu(0.10 - (1.0 - wrong_similarity)).mean()
            existing_stage28_loss = classification + logic_direction + 0.5 * cross_layer + context_flip
            preservation = _kl_preservation(full_output.logits[:, -1, :], baseline.logits[:, -1, :])
            residual_budget = torch.stack(
                [
                    torch.tensor(trace.delta_norm, device=backend.device, dtype=torch.float32)
                    for trace in full_output.traces.values()
                ]
            ).sum()
            residual_budget_loss = torch.relu(residual_budget - 0.75).square()
            hidden_norm_ratio = max(trace.hidden_norm_ratio for trace in full_output.traces.values())

            total = (
                loss_weights.existing_stage28 * existing_stage28_loss
                + loss_weights.answer_option_margin * answer_option_margin_loss
                + loss_weights.grounded_context_alignment * grounded_context_alignment_loss
                + loss_weights.wrong_context_separation * wrong_context_separation_loss
                + loss_weights.wrong_context_answer * wrong_context_answer_loss
                + loss_weights.evidence_to_answer_consistency * evidence_to_answer_consistency_loss
                + loss_weights.logit_preservation * preservation
                + loss_weights.residual_budget * residual_budget_loss
            )
            (total / gradient_accumulation).backward()
            values = {
                "existing_stage28_loss": existing_stage28_loss,
                "answer_option_margin_loss": answer_option_margin_loss,
                "grounded_context_alignment_loss": grounded_context_alignment_loss,
                "wrong_context_separation_loss": wrong_context_separation_loss,
                "wrong_context_answer_loss": wrong_context_answer_loss,
                "evidence_to_answer_consistency_loss": evidence_to_answer_consistency_loss,
                "logit_preservation_kl": preservation,
                "residual_budget_loss": residual_budget_loss,
                "correct_option_margin": 0.5 * (final_margin + update_margin),
                "total_loss": total,
            }
            for name, value in values.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu()) / gradient_accumulation
            aggregate["hidden_norm_ratio"] = max(aggregate.get("hidden_norm_ratio", 0.0), hidden_norm_ratio)

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
                f"evidence_answer_training route={route} step={step + 1}/{steps} "
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
    checkpoint_path = checkpoint_dir / f"evidence_answer_{route}_seed_{seed}.pt"
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
        "answer_option_margin_loss",
        "wrong_context_separation_loss",
        "evidence_to_answer_consistency_loss",
    )
    result = EvidenceAnswerTrainingResult(
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
