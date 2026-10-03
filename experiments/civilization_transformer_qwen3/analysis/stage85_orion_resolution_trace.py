from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage83_orion_conflict_retrieval import conflict_aware_read
from .stage84_orion_conflict_resolution import OrionConflictResolutionPolicy


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage85_orion_resolution_trace")


def resolution_aware_read(session_store: OrionMemoryStore, global_store: OrionMemoryStore, *, query: str, include_global: bool = False, limit: int = 4) -> list[dict]:
    rows = conflict_aware_read(session_store, global_store, query=query, include_global=include_global, limit=limit)
    for row in rows:
        cell = global_store.cells.get(row["cell_id"])
        row["resolution_status"] = cell.metadata.get("stage84_resolution", "unresolved") if cell is not None else "not_applicable"
        row["resolution_peer"] = cell.metadata.get("conflict_peer") if cell is not None else None
    return rows


def run_stage85_orion_resolution_trace_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    winner = global_store.write_cell(memory_system="semantic", content="approve evidence", summary="approve", source="global", confidence=0.9)
    loser = global_store.write_cell(memory_system="semantic", content="reject evidence", summary="reject", source="global", confidence=0.4)
    global_store.mark_conflict(winner.cell_id, loser.cell_id, reason="smoke")
    OrionConflictResolutionPolicy().resolve(global_store, winner.cell_id, loser.cell_id)
    rows = resolution_aware_read(session, global_store, query="evidence", include_global=True)
    summary = {"stage": "stage85_orion_resolution_trace", "rows": rows, "stage_gates": {"winner_visible": any(row["resolution_status"] == "winner" for row in rows), "loser_visible": any(row["resolution_status"] == "loser" for row in rows), "both_remain_visible": len(rows) == 2}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
