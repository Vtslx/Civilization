from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryLinkType, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage82_orion_global_maintenance")


class OrionGlobalMaintenancePolicy:
    def apply(self, store: OrionMemoryStore, *, now: float) -> dict:
        stale, conflicts = [], []
        cells = list(store.cells.values())
        for cell in cells:
            expires = cell.metadata.get("global_expires_at")
            if cell.decay_state == "active" and isinstance(expires, (int, float)) and expires <= now:
                store.update_cell(cell.cell_id, decay_state="stale", metadata={"stale_at": now})
                stale.append(cell.cell_id)
        for index, left in enumerate(cells):
            for right in cells[index + 1 :]:
                if left.metadata.get("task_name") != right.metadata.get("task_name"):
                    continue
                if {left.metadata.get("polarity"), right.metadata.get("polarity")} == {"support", "contradict"}:
                    store.mark_conflict(left.cell_id, right.cell_id, reason="stage82_opposite_polarity")
                    conflicts.append((left.cell_id, right.cell_id))
        return {"stale_cell_ids": stale, "conflict_pairs": conflicts}


def run_stage82_orion_global_maintenance_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    left = store.write_cell(memory_system="semantic", content="approve", summary="support", source="global", metadata={"task_name": "task", "polarity": "support", "global_expires_at": 200.0})
    right = store.write_cell(memory_system="semantic", content="reject", summary="contradict", source="global", metadata={"task_name": "task", "polarity": "contradict", "global_expires_at": 200.0})
    stale = store.write_cell(memory_system="semantic", content="old", summary="old", source="global", metadata={"task_name": "old", "polarity": "support", "global_expires_at": 99.0})
    result = OrionGlobalMaintenancePolicy().apply(store, now=100.0)
    active = store.read("old", limit=5)
    summary = {**store.summary(), "maintenance": result, "stage_gates": {"conflict_linked": any(link.link_type == MemoryLinkType.CONFLICT for link in store.links), "originals_preserved": left.cell_id in store.cells and right.cell_id in store.cells, "stale_excluded": stale.cell_id not in [item.cell.cell_id for item in active]}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
