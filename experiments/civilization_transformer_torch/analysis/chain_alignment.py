from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..device import resolve_device
from ..model import CivilizationAblationConfig, CivilizationTransformerTorch
from ..model.ablation import default_ablation_config
from .dataset import HARD_LOGIC_SCENARIOS, LOGIC_LABELS, LogicSample
from .hidden_states import samples_to_tensor


CHAIN_SCENARIOS = ("two_hop_logic", "three_hop_logic", "mixed_logic_priority", "ood_surface")


@dataclass(frozen=True)
class ChainStateAlignmentResult:
    seed: int
    device: str
    steps: int
    train_samples: int
    train_per_label: int
    train_batch_size: int
    train_scenarios: tuple[str, ...]
    losses: list[dict[str, float]]
    initial_total_loss: float
    final_total_loss: float
    total_loss_decreased: bool
    initial_classification_loss: float
    final_classification_loss: float
    classification_loss_decreased: bool
    initial_chain_state_loss: float
    final_chain_state_loss: float
    chain_state_loss_decreased: bool
    initial_priority_control_loss: float
    final_priority_control_loss: float
    priority_control_loss_decreased: bool
    chain_step_accuracy: float
    final_target_accuracy: float
    priority_control_accuracy: float
    disabled_losses: dict[str, bool]


def split_by_label(samples: list[LogicSample], train_per_label: int) -> tuple[list[LogicSample], list[LogicSample]]:
    train: list[LogicSample] = []
    test: list[LogicSample] = []
    for label in LOGIC_LABELS:
        label_samples = [sample for sample in samples if sample.label == label]
        if len(label_samples) <= train_per_label:
            raise ValueError("train_per_label must leave held-out samples")
        train.extend(label_samples[:train_per_label])
        test.extend(label_samples[train_per_label:])
    return train, test


def select_chain_train_samples(
    datasets: dict[str, list[LogicSample]],
    train_per_label: int,
    train_scenarios: tuple[str, ...] = HARD_LOGIC_SCENARIOS,
) -> list[LogicSample]:
    samples: list[LogicSample] = []
    for scenario in train_scenarios:
        train, _ = split_by_label(datasets[scenario], train_per_label)
        samples.extend(train)
    return samples


def labels_to_tensor(samples: list[LogicSample], device: torch.device) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(LOGIC_LABELS)}
    return torch.tensor([label_to_id[sample.label] for sample in samples], dtype=torch.long, device=device)


def scenarios_to_tensor(samples: list[LogicSample], scenario_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([scenario_to_id[sample.variant] for sample in samples], dtype=torch.long, device=device)


def _centroid_separation_loss(vectors: torch.Tensor, labels: torch.Tensor, margin: float = 4.0) -> torch.Tensor:
    centers: list[torch.Tensor] = []
    within_terms: list[torch.Tensor] = []
    for label_id in range(len(LOGIC_LABELS)):
        label_vectors = vectors[labels == label_id]
        if label_vectors.numel() == 0:
            raise ValueError(f"missing vectors for label id {label_id}")
        center = label_vectors.mean(dim=0)
        centers.append(center)
        within_terms.append(torch.mean(torch.sum((label_vectors - center) ** 2, dim=1)))
    centers_tensor = torch.stack(centers)
    distances = torch.cdist(centers_tensor, centers_tensor, p=2)
    mask = ~torch.eye(len(LOGIC_LABELS), dtype=torch.bool, device=vectors.device)
    return torch.stack(within_terms).mean() + F.relu(margin - distances[mask]).pow(2).mean()


def _hard_negative_loss(vectors: torch.Tensor, labels: torch.Tensor, margin: float = 6.0) -> torch.Tensor:
    distances = torch.cdist(vectors, vectors, p=2)
    different = labels.unsqueeze(0) != labels.unsqueeze(1)
    if not different.any():
        return torch.zeros((), dtype=vectors.dtype, device=vectors.device)
    return F.relu(margin - distances[different]).pow(2).mean()


def _balanced_indices(
    samples: list[LogicSample],
    label_ids: torch.Tensor,
    scenario_ids: torch.Tensor,
    train_scenarios: tuple[str, ...],
    scenario_to_id: dict[str, int],
    step: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    per_bucket = max(1, batch_size // (len(LOGIC_LABELS) * len(train_scenarios)))
    chunks: list[torch.Tensor] = []
    for scenario in train_scenarios:
        scenario_id = scenario_to_id[scenario]
        for label_id in range(len(LOGIC_LABELS)):
            group = torch.where((scenario_ids == scenario_id) & (label_ids == label_id))[0]
            if len(group) == 0:
                continue
            start = (step * per_bucket + label_id * 11 + scenario_id * 5) % len(group)
            local = torch.arange(start, start + per_bucket, device=device) % len(group)
            chunks.append(group[local])
    if not chunks:
        raise ValueError("empty chain training batch")
    _ = samples
    return torch.cat(chunks)


def run_chain_state_alignment_training(
    model: CivilizationTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    device: str | torch.device | None,
    seed: int,
    train_per_label: int = 80,
    steps: int = 80,
    learning_rate: float = 0.01,
    batch_size: int = 600,
    forward_kwargs: dict | None = None,
    train_scenarios: tuple[str, ...] = HARD_LOGIC_SCENARIOS,
    ablation_config: CivilizationAblationConfig | None = None,
) -> ChainStateAlignmentResult:
    target_device = resolve_device(device)
    ablation = default_ablation_config(ablation_config)
    torch.manual_seed(seed)
    model.to(target_device)
    model.train()

    train_samples = select_chain_train_samples(datasets, train_per_label, train_scenarios)
    input_ids = samples_to_tensor(train_samples, target_device)
    label_ids = labels_to_tensor(train_samples, target_device)
    scenario_to_id = {scenario: index for index, scenario in enumerate(HARD_LOGIC_SCENARIOS)}
    scenario_ids = scenarios_to_tensor(train_samples, scenario_to_id, target_device)
    chain_mask = torch.tensor([sample.variant in CHAIN_SCENARIOS for sample in train_samples], dtype=torch.bool, device=target_device)
    priority_mask = torch.tensor([sample.variant == "mixed_logic_priority" for sample in train_samples], dtype=torch.bool, device=target_device)

    classifier = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    chain_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    final_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    priority_head = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()) + list(chain_head.parameters()) + list(final_head.parameters()) + list(priority_head.parameters()),
        lr=learning_rate,
        weight_decay=0.0,
    )

    losses: list[dict[str, float]] = []
    last_chain_accuracy = 0.0
    last_final_accuracy = 0.0
    last_priority_accuracy = 0.0
    for step in range(steps + 1):
        indices = _balanced_indices(train_samples, label_ids, scenario_ids, train_scenarios, scenario_to_id, step, batch_size, target_device)
        batch_input_ids = input_ids[indices]
        batch_label_ids = label_ids[indices]
        batch_chain_mask = chain_mask[indices]
        batch_priority_mask = priority_mask[indices]

        optimizer.zero_grad(set_to_none=True)
        output = model(batch_input_ids, **(forward_kwargs or {}), ablation_config=ablation)
        pooled = output.hidden_states[-1].mean(dim=1)
        mid_layer = output.hidden_states[max(1, len(output.hidden_states) // 2)].mean(dim=1)

        next_token = F.cross_entropy(output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]), batch_input_ids[:, 1:].reshape(-1))
        classification = F.cross_entropy(classifier(pooled), batch_label_ids)
        contrastive = _centroid_separation_loss(pooled, batch_label_ids) if ablation.use_centroid_separation_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
        hard_negative = _hard_negative_loss(pooled, batch_label_ids) if ablation.use_hard_negative_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
        if batch_chain_mask.any():
            chain_logits = chain_head(mid_layer[batch_chain_mask])
            final_logits = final_head(pooled[batch_chain_mask])
            chain_state = F.cross_entropy(chain_logits, batch_label_ids[batch_chain_mask]) if ablation.use_chain_state_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
            final_state = F.cross_entropy(final_logits, batch_label_ids[batch_chain_mask]) if ablation.use_final_state_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
            last_chain_accuracy = float((chain_logits.argmax(dim=1) == batch_label_ids[batch_chain_mask]).float().mean().detach().cpu().item())
            last_final_accuracy = float((final_logits.argmax(dim=1) == batch_label_ids[batch_chain_mask]).float().mean().detach().cpu().item())
        else:
            chain_state = torch.zeros((), dtype=pooled.dtype, device=target_device)
            final_state = torch.zeros((), dtype=pooled.dtype, device=target_device)
        if batch_priority_mask.any():
            priority_logits = priority_head(pooled[batch_priority_mask])
            priority_control = F.cross_entropy(priority_logits, batch_label_ids[batch_priority_mask]) if ablation.use_priority_control_loss else torch.zeros((), dtype=pooled.dtype, device=target_device)
            last_priority_accuracy = float((priority_logits.argmax(dim=1) == batch_label_ids[batch_priority_mask]).float().mean().detach().cpu().item())
        else:
            priority_control = torch.zeros((), dtype=pooled.dtype, device=target_device)

        total = 0.10 * next_token + classification + 0.30 * contrastive + 0.06 * hard_negative + 0.65 * chain_state + 0.45 * final_state + 0.50 * priority_control
        losses.append(
            {
                "step": float(step),
                "next_token_loss": float(next_token.detach().cpu().item()),
                "classification_loss": float(classification.detach().cpu().item()),
                "contrastive_loss": float(contrastive.detach().cpu().item()),
                "hard_negative_loss": float(hard_negative.detach().cpu().item()),
                "chain_state_loss": float(chain_state.detach().cpu().item()),
                "final_state_loss": float(final_state.detach().cpu().item()),
                "priority_control_loss": float(priority_control.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            }
        )
        if step < steps:
            total.backward()
            optimizer.step()

    return ChainStateAlignmentResult(
        seed=seed,
        device=str(target_device),
        steps=steps,
        train_samples=len(train_samples),
        train_per_label=train_per_label,
        train_batch_size=min(batch_size, len(train_samples)),
        train_scenarios=train_scenarios,
        losses=losses,
        initial_total_loss=losses[0]["total_loss"],
        final_total_loss=losses[-1]["total_loss"],
        total_loss_decreased=losses[-1]["total_loss"] < losses[0]["total_loss"],
        initial_classification_loss=losses[0]["classification_loss"],
        final_classification_loss=losses[-1]["classification_loss"],
        classification_loss_decreased=losses[-1]["classification_loss"] < losses[0]["classification_loss"],
        initial_chain_state_loss=losses[0]["chain_state_loss"],
        final_chain_state_loss=losses[-1]["chain_state_loss"],
        chain_state_loss_decreased=losses[-1]["chain_state_loss"] < losses[0]["chain_state_loss"],
        initial_priority_control_loss=losses[0]["priority_control_loss"],
        final_priority_control_loss=losses[-1]["priority_control_loss"],
        priority_control_loss_decreased=losses[-1]["priority_control_loss"] < losses[0]["priority_control_loss"],
        chain_step_accuracy=last_chain_accuracy,
        final_target_accuracy=last_final_accuracy,
        priority_control_accuracy=last_priority_accuracy,
        disabled_losses={
            "chain_state_loss": not ablation.use_chain_state_loss,
            "final_state_loss": not ablation.use_final_state_loss,
            "priority_control_loss": not ablation.use_priority_control_loss,
            "centroid_separation_loss": not ablation.use_centroid_separation_loss,
            "hard_negative_loss": not ablation.use_hard_negative_loss,
        },
    )
