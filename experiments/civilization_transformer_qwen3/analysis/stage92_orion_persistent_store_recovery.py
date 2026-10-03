from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage91_orion_persistent_global_store import OrionPersistentGlobalStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage92_orion_persistent_store_recovery")


def recover_global_store(path: str | Path) -> tuple[OrionMemoryStore, dict]:
    persistence = OrionPersistentGlobalStore(path)
    try:
        return persistence.load(), {"recovered": True, "reason": None, "path": str(path)}
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        return OrionMemoryStore(), {"recovered": False, "reason": type(error).__name__, "path": str(path)}


def run_stage92_orion_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    damaged = output / "damaged_global_store.json"; damaged.write_text("{not-json", encoding="utf-8")
    store, status = recover_global_store(damaged)
    summary = {"stage": "stage92_orion_persistent_store_recovery", "status": status, "stage_gates": {"corruption_detected": status["recovered"] is False, "original_preserved": damaged.read_text(encoding="utf-8") == "{not-json", "isolated_empty_store": store.summary()["link_count"] == 0 and sum(store.summary()["cell_counts"].values()) == 0}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
