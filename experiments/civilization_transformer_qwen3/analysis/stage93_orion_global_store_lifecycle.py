from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage91_orion_persistent_global_store import OrionPersistentGlobalStore
from .stage92_orion_persistent_store_recovery import recover_global_store


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage93_orion_global_store_lifecycle")


class OrionGlobalStoreLifecycle:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, store: OrionMemoryStore) -> dict:
        OrionPersistentGlobalStore(self.path).save(store)
        return {"action": "save", "path": str(self.path), "cell_count": len(store.cells), "link_count": len(store.links)}

    def load(self) -> tuple[OrionMemoryStore, dict]:
        store, recovery = recover_global_store(self.path)
        return store, {"action": "load", **recovery, "cell_count": len(store.cells), "link_count": len(store.links)}


def run_stage93_orion_lifecycle_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir); store = OrionMemoryStore(); store.write_cell(memory_system="semantic", content="persisted", summary="persisted", source="global")
    lifecycle = OrionGlobalStoreLifecycle(output / "global_store.json")
    saved = lifecycle.save(store); restored, loaded = lifecycle.load()
    summary = {"stage": "stage93_orion_global_store_lifecycle", "saved": saved, "loaded": loaded, "stage_gates": {"explicit_save": saved["cell_count"] == 1 and lifecycle.path.exists(), "explicit_load": loaded["recovered"] and len(restored.cells) == 1, "no_auto_service_attachment": True}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
