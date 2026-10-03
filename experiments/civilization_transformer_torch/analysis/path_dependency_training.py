from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..device import resolve_device
from ..memory import MemoryEncoderTorch, MemoryItem
from ..model import CivilizationAblationConfig, CivilizationTransformerTorch
from ..model.ablation import default_ablation_config
from ..rules import RuleEngineTorch, RuleItem
from ..state import StateConfig
from .chain_alignment import _centroid_separation_loss, _hard_negative_loss, split_by_label
from .dataset import LOGIC_LABELS, OBFUSCATED_CONTEXT_CODES, PATH_DEPENDENCY_SCENARIOS, LogicSample
from .hidden_states import samples_to_tensor


@dataclass(frozen=True)
class PathDependencyLossConfig:
    use_memory_dependency_loss: bool = True
    use_state_dependency_loss: bool = True
    use_rule_dependency_loss: bool = True
    use_context_flip_loss: bool = True


@dataclass(frozen=True)
class PathDependencyTrainingResult:
    seed: int
    device: str
    steps: int
    train_samples: int
    train_per_label: int
    losses: list[dict[str, float]]
    initial_total_loss: float
    final_total_loss: float
    total_loss_decreased: bool
    initial_classification_loss: float
    final_classification_loss: float
    classification_loss_decreased: bool
    initial_memory_dependency_loss: float
    final_memory_dependency_loss: float
    memory_dependency_loss_decreased: bool
    initial_state_dependency_loss: float
    final_state_dependency_loss: float
    state_dependency_loss_decreased: bool
    initial_rule_dependency_loss: float
    final_rule_dependency_loss: float
    rule_dependency_loss_decreased: bool
    initial_context_flip_loss: float
    final_context_flip_loss: float
    context_flip_loss_decreased: bool
    disabled_losses: dict[str, bool]


LABEL_TO_ID = {label: index for index, label in enumerate(LOGIC_LABELS)}


def select_dependency_train_samples(
    datasets: dict[str, list[LogicSample]],
    train_per_label: int,
    scenarios: tuple[str, ...] = PATH_DEPENDENCY_SCENARIOS,
) -> list[LogicSample]:
    samples: list[LogicSample] = []
    for scenario in scenarios:
        train, _ = split_by_label(datasets[scenario], train_per_label)
        samples.extend(train)
    return samples


def dependency_state_for_label(label: str) -> StateConfig:
    profiles = {
        "causality": StateConfig(rigor=0.1, creativity=0.1, defensiveness=0.1),
        "negation": StateConfig(rigor=0.9, creativity=0.1, defensiveness=0.1),
        "conflict": StateConfig(rigor=0.1, creativity=0.9, defensiveness=0.1),
        "priority": StateConfig(rigor=0.1, creativity=0.1, defensiveness=0.9),
        "condition": StateConfig(rigor=0.9, creativity=0.9, defensiveness=0.1),
    }
    return profiles[label]


def dependency_context_for_sample(
    sample: LogicSample,
    model_dim: int,
    device: torch.device,
    ablation_config: CivilizationAblationConfig | None = None,
) -> dict:
    ablation = default_ablation_config(ablation_config)
    memory_vectors = torch.zeros((0, model_dim), dtype=torch.float32, device=device)
    rule_vectors = torch.zeros((0, model_dim), dtype=torch.float32, device=device)
    state = dependency_state_for_label(sample.label) if "state" in sample.required_paths else StateConfig(0.5, 0.5, 0.5)
    if "memory" in sample.required_paths and ablation.use_memory_path:
        memory_items = [
            MemoryItem(
                id=f"mem_{sample.surface_group_id}_{sample.memory_target}",
                summary=f"context memory {sample.memory_target}",
                content=f"{sample.memory_target} hidden chain resolves through selected context",
                relation_type=sample.memory_target,
                priority=1.0,
                confidence=1.0,
            )
        ]
        for index in range(sample.context_noise_count):
            memory_items.append(
                MemoryItem(
                    id=f"noise_mem_{sample.surface_group_id}_{index}",
                    summary=f"noise memory ctx_noise_{index}",
                    content=f"irrelevant path marker ctx_noise_{index} should not decide target",
                    relation_type="noise",
                    priority=0.25,
                    confidence=0.35,
                )
            )
        for index, candidate in enumerate(_conflict_candidates(sample.label, sample.conflict_context_count)):
            candidate_code = OBFUSCATED_CONTEXT_CODES[candidate] if sample.stress_profile != "direct_v1" else candidate
            memory_items.append(
                MemoryItem(
                    id=f"conflict_mem_{sample.surface_group_id}_{index}",
                    summary=f"conflict memory memory_selects_{candidate_code}",
                    content=f"conflicting path memory_selects_{candidate_code} should be downgraded",
                    relation_type=f"memory_selects_{candidate_code}",
                    priority=0.35,
                    confidence=0.45,
                )
            )
        memory_vectors = MemoryEncoderTorch(model_dim, device=device).encode(memory_items)
    if "rule" in sample.required_paths and ablation.use_rule_path:
        rules = [
            RuleItem(
                id=f"rule_{sample.surface_group_id}_{sample.rule_target}",
                type="soft",
                condition="context",
                effect=f"{sample.rule_target} choose selected context",
                priority=1.0,
                source="path_dependency",
            )
        ]
        for index in range(sample.context_noise_count):
            rules.append(
                RuleItem(
                    id=f"noise_rule_{sample.surface_group_id}_{index}",
                    type="soft",
                    condition="context",
                    effect=f"noise_rule_{index} ignore irrelevant preference",
                    priority=0.25,
                    source="path_dependency_noise",
                )
            )
        for index, candidate in enumerate(_conflict_candidates(sample.label, sample.conflict_context_count)):
            candidate_code = OBFUSCATED_CONTEXT_CODES[candidate] if sample.stress_profile != "direct_v1" else candidate
            rules.append(
                RuleItem(
                    id=f"conflict_rule_{sample.surface_group_id}_{index}",
                    type="conflict",
                    condition="context|conflict",
                    effect=f"conflicting rule_selects_{candidate_code} must be downgraded",
                    priority=0.35,
                    source="path_dependency_conflict",
                )
            )
        rule_vectors = RuleEngineTorch(
            rules,
            model_dim=model_dim,
            device=device,
        ).encode_vectors()
    return {"memory_vectors": memory_vectors, "state": state, "rule_vectors": rule_vectors}


def _conflict_candidates(label: str, count: int) -> list[str]:
    candidates = [candidate for candidate in LOGIC_LABELS if candidate != label]
    return candidates[: max(0, count)]


def labels_to_tensor(samples: list[LogicSample], device: torch.device) -> torch.Tensor:
    return torch.tensor([LABEL_TO_ID[sample.label] for sample in samples], dtype=torch.long, device=device)


def _batch_indices_by_label(samples: list[LogicSample], step: int, batch_size: int) -> list[int]:
    per_label = max(1, batch_size // len(LOGIC_LABELS))
    indices: list[int] = []
    for label in LOGIC_LABELS:
        group = [index for index, sample in enumerate(samples) if sample.label == label]
        if not group:
            continue
        start = (step * per_label + LABEL_TO_ID[label] * 13) % len(group)
        indices.extend(group[(start + offset) % len(group)] for offset in range(per_label))
    return indices


def _group_by_context(samples: list[LogicSample]) -> dict[tuple[str, str, tuple[str, ...]], list[tuple[int, LogicSample]]]:
    groups: dict[tuple[str, str, tuple[str, ...]], list[tuple[int, LogicSample]]] = {}
    for index, sample in enumerate(samples):
        key = (sample.variant, sample.label, sample.required_paths, sample.stress_profile, sample.context_noise_count, sample.conflict_context_count)
        groups.setdefault(key, []).append((index, sample))
    return groups


def contextual_forward(
    model: CivilizationTransformerTorch,
    samples: list[LogicSample],
    device: torch.device,
    ablation_config: CivilizationAblationConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    pooled_by_index: dict[int, torch.Tensor] = {}
    logits_by_index: dict[int, torch.Tensor] = {}
    traces: list = []
    for indexed_group in _group_by_context(samples).values():
        indices = [index for index, _ in indexed_group]
        group = [sample for _, sample in indexed_group]
        context = dependency_context_for_sample(group[0], model.config.model_dim, device, ablation_config)
        output = model(samples_to_tensor(group, device), **context, ablation_config=ablation_config)
        pooled = output.hidden_states[-1].mean(dim=1)
        for local_index, original_index in enumerate(indices):
            pooled_by_index[original_index] = pooled[local_index]
            logits_by_index[original_index] = output.logits[local_index]
        traces.append(output.civilization_traces)
    pooled_ordered = torch.stack([pooled_by_index[index] for index in range(len(samples))], dim=0)
    logits_ordered = torch.stack([logits_by_index[index] for index in range(len(samples))], dim=0)
    return pooled_ordered, logits_ordered, traces


def run_path_dependency_training(
    model: CivilizationTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    device: str | torch.device | None,
    seed: int,
    train_per_label: int = 80,
    steps: int = 60,
    learning_rate: float = 0.01,
    batch_size: int = 300,
    ablation_config: CivilizationAblationConfig | None = None,
    loss_config: PathDependencyLossConfig | None = None,
) -> PathDependencyTrainingResult:
    target_device = resolve_device(device)
    ablation = default_ablation_config(ablation_config)
    losses_enabled = loss_config or PathDependencyLossConfig()
    torch.manual_seed(seed)
    model.to(target_device)
    model.train()
    train_samples = select_dependency_train_samples(datasets, train_per_label)
    classifier = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    memory_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    state_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    rule_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    flip_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()) + list(memory_head.parameters()) + list(state_head.parameters()) + list(rule_head.parameters()) + list(flip_head.parameters()),
        lr=learning_rate,
        weight_decay=0.0,
    )
    losses: list[dict[str, float]] = []
    for step in range(steps + 1):
        batch = [train_samples[index] for index in _batch_indices_by_label(train_samples, step, batch_size)]
        labels = labels_to_tensor(batch, target_device)
        optimizer.zero_grad(set_to_none=True)
        pooled, logits, _ = contextual_forward(model, batch, target_device, ablation)
        input_ids = samples_to_tensor(batch, target_device)
        next_token = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
        classification = F.cross_entropy(classifier(pooled), labels)
        contrastive = _centroid_separation_loss(pooled, labels) if ablation.use_centroid_separation_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
        hard_negative = _hard_negative_loss(pooled, labels) if ablation.use_hard_negative_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
        memory_mask = torch.tensor(["memory" in sample.required_paths for sample in batch], dtype=torch.bool, device=target_device)
        state_mask = torch.tensor(["state" in sample.required_paths for sample in batch], dtype=torch.bool, device=target_device)
        rule_mask = torch.tensor(["rule" in sample.required_paths for sample in batch], dtype=torch.bool, device=target_device)
        flip_mask = torch.tensor([sample.variant == "surface_invariant_label_flip" for sample in batch], dtype=torch.bool, device=target_device)
        memory_loss = F.cross_entropy(memory_head(pooled[memory_mask]), labels[memory_mask]) if losses_enabled.use_memory_dependency_loss and memory_mask.any() else torch.zeros((), dtype=pooled.dtype, device=target_device)
        state_loss = F.cross_entropy(state_head(pooled[state_mask]), labels[state_mask]) if losses_enabled.use_state_dependency_loss and state_mask.any() else torch.zeros((), dtype=pooled.dtype, device=target_device)
        rule_loss = F.cross_entropy(rule_head(pooled[rule_mask]), labels[rule_mask]) if losses_enabled.use_rule_dependency_loss and rule_mask.any() else torch.zeros((), dtype=pooled.dtype, device=target_device)
        flip_loss = F.cross_entropy(flip_head(pooled[flip_mask]), labels[flip_mask]) if losses_enabled.use_context_flip_loss and flip_mask.any() else torch.zeros((), dtype=pooled.dtype, device=target_device)
        total = 0.05 * next_token + classification + 0.25 * contrastive + 0.05 * hard_negative + 0.70 * memory_loss + 0.70 * state_loss + 0.70 * rule_loss + 0.80 * flip_loss
        losses.append(
            {
                "step": float(step),
                "next_token_loss": float(next_token.detach().cpu().item()),
                "classification_loss": float(classification.detach().cpu().item()),
                "contrastive_loss": float(contrastive.detach().cpu().item()),
                "hard_negative_loss": float(hard_negative.detach().cpu().item()),
                "memory_dependency_loss": float(memory_loss.detach().cpu().item()),
                "state_dependency_loss": float(state_loss.detach().cpu().item()),
                "rule_dependency_loss": float(rule_loss.detach().cpu().item()),
                "context_flip_loss": float(flip_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            }
        )
        if step < steps:
            total.backward()
            optimizer.step()

    def decreased(key: str) -> bool:
        return losses[-1][key] < losses[0][key]

    return PathDependencyTrainingResult(
        seed=seed,
        device=str(target_device),
        steps=steps,
        train_samples=len(train_samples),
        train_per_label=train_per_label,
        losses=losses,
        initial_total_loss=losses[0]["total_loss"],
        final_total_loss=losses[-1]["total_loss"],
        total_loss_decreased=decreased("total_loss"),
        initial_classification_loss=losses[0]["classification_loss"],
        final_classification_loss=losses[-1]["classification_loss"],
        classification_loss_decreased=decreased("classification_loss"),
        initial_memory_dependency_loss=losses[0]["memory_dependency_loss"],
        final_memory_dependency_loss=losses[-1]["memory_dependency_loss"],
        memory_dependency_loss_decreased=decreased("memory_dependency_loss"),
        initial_state_dependency_loss=losses[0]["state_dependency_loss"],
        final_state_dependency_loss=losses[-1]["state_dependency_loss"],
        state_dependency_loss_decreased=decreased("state_dependency_loss"),
        initial_rule_dependency_loss=losses[0]["rule_dependency_loss"],
        final_rule_dependency_loss=losses[-1]["rule_dependency_loss"],
        rule_dependency_loss_decreased=decreased("rule_dependency_loss"),
        initial_context_flip_loss=losses[0]["context_flip_loss"],
        final_context_flip_loss=losses[-1]["context_flip_loss"],
        context_flip_loss_decreased=decreased("context_flip_loss"),
        disabled_losses={
            "memory_dependency_loss": not losses_enabled.use_memory_dependency_loss,
            "state_dependency_loss": not losses_enabled.use_state_dependency_loss,
            "rule_dependency_loss": not losses_enabled.use_rule_dependency_loss,
            "context_flip_loss": not losses_enabled.use_context_flip_loss,
        },
    )


@torch.no_grad()
def contextual_vectors(
    model: CivilizationTransformerTorch,
    samples: list[LogicSample],
    device: str | torch.device | None,
    ablation_config: CivilizationAblationConfig | None = None,
) -> tuple[np.ndarray, list]:
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    vectors, _, traces = contextual_forward(model, samples, target_device, ablation_config)
    if not torch.isfinite(vectors).all():
        raise ValueError("path dependency hidden states contain NaN or Inf")
    return vectors.detach().cpu().numpy(), traces
