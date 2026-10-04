from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..device import resolve_device
from ..model import MiniTransformerTorch
from .dataset import LOGIC_LABELS, LOGIC_VARIANTS, LogicSample
from .hidden_states import samples_to_tensor


@dataclass(frozen=True)
class AlignmentTrainingResult:
    seed: int
    device: str
    steps: int
    batch_shape: tuple[int, int]
    train_samples: int
    train_per_label: int
    losses: list[dict[str, float]]
    initial_total_loss: float
    final_total_loss: float
    total_loss_decreased: bool
    initial_classification_loss: float
    final_classification_loss: float
    classification_loss_decreased: bool


def select_alignment_train_samples(
    datasets: dict[str, list[LogicSample]],
    train_per_label: int,
) -> list[LogicSample]:
    samples: list[LogicSample] = []
    for variant in LOGIC_VARIANTS:
        if variant not in datasets:
            raise ValueError(f"missing variant {variant}")
        variant_samples = datasets[variant]
        for label in LOGIC_LABELS:
            label_samples = [sample for sample in variant_samples if sample.label == label]
            if len(label_samples) <= train_per_label:
                raise ValueError("train_per_label must leave held-out samples for every label and variant")
            samples.extend(label_samples[:train_per_label])
    return samples


def _labels_to_tensor(samples: list[LogicSample], device: torch.device) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(LOGIC_LABELS)}
    return torch.tensor([label_to_id[sample.label] for sample in samples], dtype=torch.long, device=device)


def _pooled_final_hidden(model: MiniTransformerTorch, input_ids: torch.Tensor, forward_kwargs: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(input_ids, **(forward_kwargs or {}))
    pooled = output.hidden_states[-1].mean(dim=1)
    return output.logits, pooled


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
    separation_penalty = F.relu(margin - distances[mask]).pow(2).mean()
    within_loss = torch.stack(within_terms).mean()
    return within_loss + separation_penalty


def run_alignment_training(
    model: MiniTransformerTorch,
    datasets: dict[str, list[LogicSample]],
    device: str | torch.device | None,
    seed: int,
    train_per_label: int = 40,
    steps: int = 120,
    learning_rate: float = 0.012,
    next_token_weight: float = 0.15,
    classification_weight: float = 1.0,
    contrastive_weight: float = 0.35,
    margin: float = 4.0,
    forward_kwargs: dict | None = None,
) -> AlignmentTrainingResult:
    target_device = resolve_device(device)
    torch.manual_seed(seed)
    if target_device.type == "mps":
        torch.mps.manual_seed(seed)

    model.to(target_device)
    model.train()
    train_samples = select_alignment_train_samples(datasets, train_per_label=train_per_label)
    input_ids = samples_to_tensor(train_samples, target_device)
    label_ids = _labels_to_tensor(train_samples, target_device)
    classifier = nn.Linear(model.config.model_dim, len(LOGIC_LABELS)).to(target_device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=0.0,
    )

    losses: list[dict[str, float]] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, pooled = _pooled_final_hidden(model, input_ids, forward_kwargs=forward_kwargs)
        next_token = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
        classification = F.cross_entropy(classifier(pooled), label_ids)
        contrastive = _centroid_separation_loss(pooled, label_ids, margin=margin)
        total = next_token_weight * next_token + classification_weight * classification + contrastive_weight * contrastive
        losses.append(
            {
                "step": float(step),
                "next_token_loss": float(next_token.detach().cpu().item()),
                "classification_loss": float(classification.detach().cpu().item()),
                "contrastive_loss": float(contrastive.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            }
        )
        if step < steps:
            total.backward()
            optimizer.step()

    return AlignmentTrainingResult(
        seed=seed,
        device=str(target_device),
        steps=steps,
        batch_shape=tuple(input_ids.shape),
        train_samples=len(train_samples),
        train_per_label=train_per_label,
        losses=losses,
        initial_total_loss=losses[0]["total_loss"],
        final_total_loss=losses[-1]["total_loss"],
        total_loss_decreased=losses[-1]["total_loss"] < losses[0]["total_loss"],
        initial_classification_loss=losses[0]["classification_loss"],
        final_classification_loss=losses[-1]["classification_loss"],
        classification_loss_decreased=losses[-1]["classification_loss"] < losses[0]["classification_loss"],
    )
