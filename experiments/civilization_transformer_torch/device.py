from __future__ import annotations

import torch


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
