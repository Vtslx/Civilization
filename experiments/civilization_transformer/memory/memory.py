from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class MemoryItem:
    id: str
    summary: str
    content: str
    relation_type: str
    priority: float
    confidence: float
    embedding: np.ndarray | None = None


class MemoryEncoder:
    def __init__(self, model_dim: int, seed: int = 17):
        self.model_dim = model_dim
        self.rng = np.random.default_rng(seed)

    def _hash_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.model_dim, dtype=float)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.model_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode_item(self, item: MemoryItem) -> np.ndarray:
        if item.embedding is not None:
            base = np.asarray(item.embedding, dtype=float)
            if base.shape != (self.model_dim,):
                raise ValueError("MemoryItem.embedding must match model_dim")
        else:
            base = self._hash_embedding(f"{item.summary} {item.content} {item.relation_type}")
        weight = max(item.priority, 0.0) * max(min(item.confidence, 1.0), 0.0)
        return base * weight

    def encode(self, items: list[MemoryItem]) -> np.ndarray:
        if not items:
            return np.zeros((0, self.model_dim), dtype=float)
        return np.stack([self.encode_item(item) for item in items], axis=0)
