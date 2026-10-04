from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from .stage73_orion_memory_kernel import MemoryReadResult


@dataclass(frozen=True)
class OrionAdapterMemoryItem:
    cell_id: str
    memory_system: str
    source: str
    retrieval_score: float
    text: str
    truncated: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "cell_id": self.cell_id,
            "memory_system": self.memory_system,
            "source": self.source,
            "retrieval_score": self.retrieval_score,
            "text": self.text,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class OrionEncodedAdapterMemory:
    items: tuple[OrionAdapterMemoryItem, ...]
    vectors: torch.Tensor
    mask: torch.Tensor


class TextGroupEncoder(Protocol):
    def encode_text_groups(self, groups: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class OrionAdapterContextBridge:
    def __init__(self, *, max_items: int = 4, max_item_chars: int = 512) -> None:
        if max_items < 0:
            raise ValueError("max_items must be >= 0")
        if max_item_chars < 32:
            raise ValueError("max_item_chars must be >= 32")
        self.max_items = max_items
        self.max_item_chars = max_item_chars

    def items_for_results(self, results: Sequence[MemoryReadResult], *, limit: int | None = None) -> tuple[OrionAdapterMemoryItem, ...]:
        selected_limit = self.max_items if limit is None else min(self.max_items, limit)
        if selected_limit < 0:
            raise ValueError("limit must be >= 0")
        if selected_limit == 0:
            return ()
        ordered = sorted(results, key=lambda result: (-result.score, result.cell.time_index, result.cell.cell_id))
        items: list[OrionAdapterMemoryItem] = []
        seen: set[str] = set()
        for result in ordered:
            cell = result.cell
            if cell.cell_id in seen or cell.decay_state != "active":
                continue
            seen.add(cell.cell_id)
            prefix = f"[orion system={cell.memory_system.value} cell_id={cell.cell_id}] "
            body = f"{cell.summary}: {cell.content}".strip()
            text = prefix + body
            truncated = len(text) > self.max_item_chars
            if truncated:
                text = text[: self.max_item_chars]
            items.append(
                OrionAdapterMemoryItem(
                    cell_id=cell.cell_id,
                    memory_system=cell.memory_system.value,
                    source=cell.source,
                    retrieval_score=result.score,
                    text=text,
                    truncated=truncated,
                )
            )
            if len(items) >= selected_limit:
                break
        return tuple(items)

    def encode(self, encoder: TextGroupEncoder, items: Sequence[OrionAdapterMemoryItem]) -> OrionEncodedAdapterMemory:
        vectors, mask = encoder.encode_text_groups([[item.text for item in items]])
        if vectors.ndim != 3 or vectors.shape[0] != 1 or vectors.shape[1] != len(items):
            raise ValueError("adapter memory vectors must have shape [1, item_count, hidden_size]")
        if mask.shape != vectors.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("adapter memory mask must match vectors and use bool dtype")
        if not torch.isfinite(vectors).all():
            raise ValueError("adapter memory vectors contain NaN or Inf")
        if len(items) and not bool(mask.all()):
            raise ValueError("adapter memory mask omitted a serialized item")
        return OrionEncodedAdapterMemory(items=tuple(items), vectors=vectors, mask=mask)


def bridge_memory_item_text(results: Sequence[MemoryReadResult], *, max_items: int, max_item_chars: int, limit: int | None = None) -> tuple[OrionAdapterMemoryItem, ...]:
    return OrionAdapterContextBridge(max_items=max_items, max_item_chars=max_item_chars).items_for_results(results, limit=limit)
