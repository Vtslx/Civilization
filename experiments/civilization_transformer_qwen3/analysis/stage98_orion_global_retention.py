from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage98_orion_global_retention")


class OrionGlobalRetentionPolicy:
    def __init__(self, *, max_active_cells: int = 100) -> None:
        if max_active_cells < 1:
            raise ValueError("max_active_cells must be >= 1")
        self.max_active_cells = max_active_cells

    def apply(self, store: OrionMemoryStore) -> dict:
        active = [cell for cell in store.cells.values() if cell.decay_state == "active"]
        active.sort(key=lambda cell: (-cell.importance, -cell.confidence, -cell.time_index, cell.cell_id))
        retired = []
        for cell in active[self.max_active_cells :]:
            store.update_cell(cell.cell_id, decay_state="retired", metadata={"retention_reason": "active_capacity"})
            retired.append(cell.cell_id)
        return {"retired_cell_ids": retired, "active_before": len(active), "active_after": len(active) - len(retired)}


def run_stage98_orion_retention_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore()
    high = store.write_cell(memory_system="semantic", content="high", summary="high", source="global", importance=1.0)
    low = store.write_cell(memory_system="semantic", content="low", summary="low", source="global", importance=0.1)
    result = OrionGlobalRetentionPolicy(max_active_cells=1).apply(store)
    summary = {**store.summary(), "retention": result, "stage_gates": {"low_retired": low.cell_id in result["retired_cell_ids"], "high_active": store.cells[high.cell_id].decay_state == "active", "retired_preserved": low.cell_id in store.cells, "retired_excluded": not store.read("low", limit=5)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); store.write_artifacts(output_dir, summary=summary)
    return summary
