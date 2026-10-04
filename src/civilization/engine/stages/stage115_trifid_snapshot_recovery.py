from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage114_trifid_episode_snapshot import TrifidEpisodeSnapshot
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage115_trifid_snapshot_recovery")


def recover_trifid_episode_snapshot(path: str | Path, store: OrionMemoryStore) -> tuple[TrifidEpisodeBinder, dict[str, Any]]:
    snapshot = TrifidEpisodeSnapshot(path)
    try:
        binder = snapshot.load(store)
        return binder, {"recovered": True, "reason": None, "path": str(snapshot.path)}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return TrifidEpisodeBinder(), {"recovered": False, "reason": type(error).__name__, "path": str(snapshot.path)}


def run_stage115_trifid_snapshot_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="observe", action="inspect", outcome="secured")
    valid_path = output / "valid.json"
    TrifidEpisodeSnapshot(valid_path).save(binder)
    restored, valid_status = recover_trifid_episode_snapshot(valid_path, store)
    corrupt_path = output / "corrupt.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not-json", encoding="utf-8")
    before_cells, before_links = len(store.cells), len(store.links)
    corrupt, corrupt_status = recover_trifid_episode_snapshot(corrupt_path, store)
    dangling_path = output / "dangling.json"
    dangling_path.write_text(json.dumps({"version": 1, "next_episode": 2, "frames": [{**frame.to_dict(), "anchor_cell_id": "episodic-missing"}]}), encoding="utf-8")
    dangling, dangling_status = recover_trifid_episode_snapshot(dangling_path, store)
    summary = {
        "stage": "stage115_trifid_snapshot_recovery",
        "stage_gates": {
            "valid_snapshot_recovered": valid_status["recovered"] and frame.episode_id in restored.frames,
            "corrupt_snapshot_failed_closed": not corrupt_status["recovered"] and corrupt_status["reason"] == "JSONDecodeError" and corrupt.frames == {},
            "dangling_snapshot_failed_closed": not dangling_status["recovered"] and dangling_status["reason"] == "ValueError" and dangling.frames == {},
            "orion_store_unchanged": len(store.cells) == before_cells and len(store.links) == before_links,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
