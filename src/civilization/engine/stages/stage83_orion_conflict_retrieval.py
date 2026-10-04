from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryLinkType, OrionMemoryStore
from .stage79_orion_global_retrieval import OrionGlobalRetrievalRouter


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage83_orion_conflict_retrieval")


def conflict_aware_read(session_store: OrionMemoryStore, global_store: OrionMemoryStore, *, query: str, include_global: bool = False, limit: int = 4) -> list[dict]:
    rows = []
    for tiered in OrionGlobalRetrievalRouter().read(session_store, global_store, query=query, include_global=include_global, limit=limit):
        cell_id = tiered.result.cell.cell_id
        conflicts = sorted({link.target_cell_id if link.source_cell_id == cell_id else link.source_cell_id for link in global_store.links if link.link_type == MemoryLinkType.CONFLICT and cell_id in {link.source_cell_id, link.target_cell_id}}) if tiered.tier == "global" else []
        rows.append({"tier": tiered.tier, "cell_id": cell_id, "score": tiered.result.score, "conflict_cell_ids": conflicts, "requires_resolution": bool(conflicts)})
    return rows


def run_stage83_orion_conflict_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    left = global_store.write_cell(memory_system="semantic", content="approve evidence", summary="approve", source="global")
    right = global_store.write_cell(memory_system="semantic", content="reject evidence", summary="reject", source="global")
    global_store.mark_conflict(left.cell_id, right.cell_id, reason="smoke")
    rows = conflict_aware_read(session, global_store, query="evidence", include_global=True)
    summary = {"stage": "stage83_orion_conflict_retrieval", "rows": rows, "stage_gates": {"both_conflicts_visible": len(rows) == 2 and all(row["requires_resolution"] for row in rows), "no_auto_resolution": all(len(row["conflict_cell_ids"]) == 1 for row in rows)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
