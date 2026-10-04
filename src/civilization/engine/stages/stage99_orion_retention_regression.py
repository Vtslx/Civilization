from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage91_orion_persistent_global_store import OrionPersistentGlobalStore
from .stage98_orion_global_retention import OrionGlobalRetentionPolicy


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage99_orion_retention_regression")


def run_stage99_orion_retention_regression(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir); store = OrionMemoryStore()
    store.write_cell(memory_system="semantic", content="high", summary="high", source="global", importance=1.0)
    low = store.write_cell(memory_system="semantic", content="low", summary="low", source="global", importance=0.1)
    OrionGlobalRetentionPolicy(max_active_cells=1).apply(store)
    path = output / "global.json"; OrionPersistentGlobalStore(path).save(store); restored = OrionPersistentGlobalStore(path).load()
    states = {cell.decay_state for cell in restored.cells.values()}
    summary = {"stage": "stage99_orion_retention_regression", "stage_gates": {"retired_persisted": restored.cells[low.cell_id].decay_state == "retired", "active_and_retired_present": {"active", "retired"} <= states}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
