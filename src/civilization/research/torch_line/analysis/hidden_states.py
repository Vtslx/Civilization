from __future__ import annotations

import torch

from ..device import resolve_device
from ..model import MiniTransformerTorch
from .dataset import LogicSample


def samples_to_tensor(samples: list[LogicSample], device: str | torch.device | None = None) -> torch.Tensor:
    target_device = resolve_device(device)
    return torch.tensor([sample.token_ids for sample in samples], dtype=torch.long, device=target_device)


@torch.no_grad()
def collect_hidden_state_representations(
    model: MiniTransformerTorch,
    samples: list[LogicSample],
    device: str | torch.device | None = None,
) -> dict[str, list[torch.Tensor]]:
    target_device = resolve_device(device)
    model = model.to(target_device)
    model.eval()
    input_ids = samples_to_tensor(samples, target_device)
    output = model(input_ids)

    mean_pool: list[torch.Tensor] = []
    last_token_pool: list[torch.Tensor] = []
    for layer_hidden in output.hidden_states:
        if not torch.isfinite(layer_hidden).all():
            raise ValueError("hidden states contain NaN or Inf")
        mean_pool.append(layer_hidden.mean(dim=1).detach().cpu())
        last_token_pool.append(layer_hidden[:, -1, :].detach().cpu())
    return {"mean": mean_pool, "last": last_token_pool}
