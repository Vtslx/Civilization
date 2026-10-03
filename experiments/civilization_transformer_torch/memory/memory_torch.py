from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from ..device import resolve_device


@dataclass(frozen=True)
class MemoryItem:
    id: str
    summary: str
    content: str
    relation_type: str
    priority: float
    confidence: float
    embedding: torch.Tensor | None = None


class MemoryEncoderTorch:
    def __init__(self, model_dim: int, device: str | torch.device | None = None):
        self.model_dim = model_dim
        self.device = resolve_device(device)

    def _hash_embedding(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.model_dim, dtype=torch.float32, device=self.device)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.model_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = torch.linalg.vector_norm(vector)
        return vector / norm if norm.item() > 0 else vector

    def encode_item(self, item: MemoryItem) -> torch.Tensor:
        if item.embedding is not None:
            base = item.embedding.to(self.device, dtype=torch.float32)
            if tuple(base.shape) != (self.model_dim,):
                raise ValueError("MemoryItem.embedding must match model_dim")
        else:
            base = self._hash_embedding(f"{item.summary} {item.content} {item.relation_type}")
        weight = max(item.priority, 0.0) * max(min(item.confidence, 1.0), 0.0)
        return base * weight

    def encode(self, items: list[MemoryItem]) -> torch.Tensor:
        if not items:
            return torch.zeros((0, self.model_dim), dtype=torch.float32, device=self.device)
        return torch.stack([self.encode_item(item) for item in items], dim=0)
