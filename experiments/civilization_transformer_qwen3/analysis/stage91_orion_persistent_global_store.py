from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryCell, MemoryLink, MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage91_orion_persistent_global_store")


class OrionPersistentGlobalStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, store: OrionMemoryStore) -> None:
        payload = {"cells": [cell.to_dict() for cell in store.cells.values()], "links": [link.to_dict() for link in store.links]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> OrionMemoryStore:
        store = OrionMemoryStore()
        if not self.path.exists():
            return store
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for item in payload.get("cells", []):
            cell = MemoryCell(**{**item, "memory_system": MemorySystem(item["memory_system"])})
            store.cells[cell.cell_id] = cell
        for item in payload.get("links", []):
            store.links.append(MemoryLink(**{**item, "link_type": MemoryLinkType(item["link_type"])}))
        store._next_cell_id = len(store.cells) + 1  # noqa: SLF001 - restore monotonic ids
        return store


def run_stage91_orion_persistence_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir); store = OrionMemoryStore()
    left = store.write_cell(memory_system="semantic", content="winner", summary="winner", source="global", metadata={"stage84_resolution": "winner"})
    right = store.write_cell(memory_system="semantic", content="loser", summary="loser", source="global")
    store.mark_conflict(left.cell_id, right.cell_id, reason="smoke")
    persistence = OrionPersistentGlobalStore(output / "global_store.json"); persistence.save(store); restored = persistence.load()
    summary = {"stage": "stage91_orion_persistence", "stage_gates": {"file_written": persistence.path.exists(), "cells_restored": set(restored.cells) == set(store.cells), "links_restored": len(restored.links) == len(store.links), "metadata_restored": restored.cells[left.cell_id].metadata.get("stage84_resolution") == "winner"}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
