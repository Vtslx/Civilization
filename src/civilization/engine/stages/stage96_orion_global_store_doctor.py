from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage93_orion_global_store_lifecycle import OrionGlobalStoreLifecycle


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage96_orion_global_store_doctor")


def inspect_or_restore_global_store(path: str | Path, *, restore: bool = False) -> tuple[OrionMemoryStore, dict]:
    lifecycle = OrionGlobalStoreLifecycle(path)
    status = {"path": str(path), "exists": Path(path).exists(), "restore_requested": restore}
    if not restore:
        return OrionMemoryStore(), {**status, "restored": False, "reason": "diagnostic_only"}
    store, loaded = lifecycle.load()
    return store, {**status, "restored": loaded["recovered"], "recovery": loaded}


def run_stage96_orion_doctor_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir); source = OrionMemoryStore(); source.write_cell(memory_system="semantic", content="persist", summary="persist", source="global")
    lifecycle = OrionGlobalStoreLifecycle(output / "global.json"); lifecycle.save(source)
    dry_store, dry = inspect_or_restore_global_store(lifecycle.path)
    restored, applied = inspect_or_restore_global_store(lifecycle.path, restore=True)
    summary = {"stage": "stage96_orion_global_store_doctor", "dry": dry, "applied": applied, "stage_gates": {"dry_does_not_load": not dry["restored"] and not dry_store.cells, "explicit_restore_loads": applied["restored"] and len(restored.cells) == 1, "file_status_visible": dry["exists"]}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
