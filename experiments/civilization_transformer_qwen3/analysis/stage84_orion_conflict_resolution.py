from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryLinkType, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage84_orion_conflict_resolution")


class OrionConflictResolutionPolicy:
    def __init__(self, *, confidence_margin: float = 0.2) -> None:
        if not 0.0 < confidence_margin <= 1.0:
            raise ValueError("confidence_margin must be in (0, 1]")
        self.confidence_margin = confidence_margin

    def resolve(self, store: OrionMemoryStore, left_id: str, right_id: str) -> dict:
        left, right = store.cells[left_id], store.cells[right_id]
        if not any(link.link_type == MemoryLinkType.CONFLICT and {link.source_cell_id, link.target_cell_id} == {left_id, right_id} for link in store.links):
            raise ValueError("cells are not linked as a conflict")
        delta = left.confidence - right.confidence
        if abs(delta) < self.confidence_margin:
            return {"resolved": False, "reason": "confidence_margin_not_met", "winner_cell_id": None, "loser_cell_id": None}
        winner, loser = (left, right) if delta > 0 else (right, left)
        store.update_cell(winner.cell_id, metadata={"stage84_resolution": "winner", "conflict_peer": loser.cell_id})
        store.update_cell(loser.cell_id, metadata={"stage84_resolution": "loser", "conflict_peer": winner.cell_id})
        return {"resolved": True, "reason": "confidence_margin_met", "winner_cell_id": winner.cell_id, "loser_cell_id": loser.cell_id}


def run_stage84_orion_resolution_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore()
    left = store.write_cell(memory_system="semantic", content="approve", summary="approve", source="global", confidence=0.9)
    right = store.write_cell(memory_system="semantic", content="reject", summary="reject", source="global", confidence=0.4)
    store.mark_conflict(left.cell_id, right.cell_id, reason="smoke")
    result = OrionConflictResolutionPolicy().resolve(store, left.cell_id, right.cell_id)
    summary = {**store.summary(), "resolution": result, "stage_gates": {"resolved": result["resolved"], "conflict_preserved": any(link.link_type == MemoryLinkType.CONFLICT for link in store.links), "both_cells_preserved": left.cell_id in store.cells and right.cell_id in store.cells, "winner_audited": store.cells[left.cell_id].metadata.get("stage84_resolution") == "winner"}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
