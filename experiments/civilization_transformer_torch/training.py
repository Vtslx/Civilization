from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from .device import resolve_device
from .model import MiniTransformerTorch, TransformerConfigTorch
from .model.transformer_torch import next_token_loss


@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float
    final_loss: float
    loss_decreased: bool
    device: str
    seed: int
    batch_shape: tuple[int, int]
    losses: list[float]


def build_next_token_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(seq_len + 1, device=device).unsqueeze(0).repeat(batch_size, 1)
    offsets = torch.arange(batch_size, device=device).unsqueeze(1)
    data = (base + offsets) % vocab_size
    return data[:, :-1].long(), data[:, 1:].long()


def run_minimal_training_loop(
    device: str | torch.device | None = None,
    seed: int = 101,
    steps: int = 45,
    learning_rate: float = 0.025,
) -> TrainingResult:
    target_device = resolve_device(device)
    torch.manual_seed(seed)
    if target_device.type == "mps":
        torch.mps.manual_seed(seed)

    config = TransformerConfigTorch(vocab_size=24, model_dim=16, hidden_dim=32, num_heads=4, num_layers=2, max_seq_len=12, seed=seed)
    model = MiniTransformerTorch(config).to(target_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    input_ids, targets = build_next_token_batch(batch_size=8, seq_len=8, vocab_size=config.vocab_size, device=target_device)

    losses: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids)
        loss = next_token_loss(output.logits, targets)
        losses.append(float(loss.detach().cpu().item()))
        if step < steps:
            loss.backward()
            optimizer.step()

    return TrainingResult(
        initial_loss=losses[0],
        final_loss=losses[-1],
        loss_decreased=losses[-1] < losses[0],
        device=str(target_device),
        seed=seed,
        batch_shape=tuple(input_ids.shape),
        losses=losses,
    )
