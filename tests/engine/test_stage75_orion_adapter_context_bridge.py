from __future__ import annotations

import torch

from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from civilization.engine.stages.stage75_orion_adapter_context_bridge import OrionAdapterContextBridge


class RecordingEncoder:
    def __init__(self) -> None:
        self.groups: list[list[str]] = []

    def encode_text_groups(self, groups: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        self.groups = groups
        count = len(groups[0])
        return torch.ones((1, count, 1024)), torch.ones((1, count), dtype=torch.bool)


def test_stage75_bridge_serializes_active_cells_deterministically() -> None:
    clock = {"now": 100.0}
    store = OrionMemoryStore(now_fn=lambda: clock["now"])
    semantic = store.write_cell(memory_system=MemorySystem.SEMANTIC, content="semantic proof", summary="semantic summary", source="test", importance=1.0)
    episodic = store.write_cell(memory_system=MemorySystem.EPISODIC, content="episodic proof", summary="episodic summary", source="test", importance=0.5)
    expired = store.write_cell(memory_system=MemorySystem.WORKING, content="expired proof", summary="expired summary", source="test", ttl_seconds=1.0, time_index=90.0)
    clock["now"] = 102.0
    results = store.read("proof", include_expired=True, limit=10)
    store.expire_working_memory()

    items = OrionAdapterContextBridge(max_items=4, max_item_chars=128).items_for_results(results)

    assert [item.cell_id for item in items] == [semantic.cell_id, episodic.cell_id]
    assert expired.cell_id not in [item.cell_id for item in items]
    assert all(f"cell_id={item.cell_id}" in item.text for item in items)


def test_stage75_bridge_encodes_vectors_with_cell_mapping() -> None:
    store = OrionMemoryStore()
    cell = store.write_cell(memory_system=MemorySystem.SEMANTIC, content="adapter context", summary="adapter summary", source="test")
    items = OrionAdapterContextBridge(max_items=2, max_item_chars=128).items_for_results(store.read("adapter", limit=2))
    encoder = RecordingEncoder()

    encoded = OrionAdapterContextBridge(max_items=2, max_item_chars=128).encode(encoder, items)

    assert encoded.vectors.shape == (1, 1, 1024)
    assert encoded.mask.dtype == torch.bool
    assert [item.cell_id for item in encoded.items] == [cell.cell_id]
    assert encoder.groups == [[items[0].text]]
