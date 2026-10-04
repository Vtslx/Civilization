from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage116_trifid_paired_checkpoint import TrifidPairedCheckpoint
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage117_trifid_checkpoint_recovery")


class TrifidCheckpointRepository:
    """Stores checkpoint generations and recovers the newest valid generation."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, generation: int, store: OrionMemoryStore, binder: TrifidEpisodeBinder) -> Path:
        if generation < 1:
            raise ValueError("generation must be >= 1")
        path = self.directory / f"checkpoint-{generation:06d}"
        TrifidPairedCheckpoint(path).save(store, binder)
        return path

    def load_latest(self) -> tuple[OrionMemoryStore, TrifidEpisodeBinder, dict[str, Any]]:
        paths = sorted((path for path in self.directory.glob("checkpoint-*") if path.is_dir()), reverse=True)
        skipped: list[str] = []
        for path in paths:
            try:
                store, binder = TrifidPairedCheckpoint(path).load()
                return store, binder, {"recovered": True, "generation": path.name, "skipped": skipped}
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped.append(path.name)
        return OrionMemoryStore(), TrifidEpisodeBinder(), {"recovered": False, "generation": None, "skipped": skipped}


def run_stage117_trifid_checkpoint_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    repo = TrifidCheckpointRepository(output / "checkpoints")
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    first = store.write_cell(memory_system="episodic", content="robot laboratory arrival", summary="arrival", source="obs", time_index=100.0)
    frame = binder.bind(store, source_cell_ids=[first.cell_id], scene="laboratory", entities=["robot"], goal="observe", action="inspect", outcome="secured")
    generation_one = repo.save(1, store, binder)
    second = store.write_cell(memory_system="episodic", content="robot laboratory exit", summary="exit", source="obs", time_index=200.0)
    binder.bind(store, source_cell_ids=[second.cell_id], scene="laboratory", entities=["robot"], goal="leave", action="exit", outcome="complete")
    generation_two = repo.save(2, store, binder)
    (generation_two / "trifid_frames.json").write_text("{}", encoding="utf-8")
    restored_store, restored_binder, status = repo.load_latest()
    summary = {
        "stage": "stage117_trifid_checkpoint_recovery",
        "stage_gates": {
            "latest_corrupt_generation_skipped": status["skipped"] == [generation_two.name],
            "previous_generation_recovered": status["recovered"] and status["generation"] == generation_one.name,
            "older_frame_restored": frame.episode_id in restored_binder.frames and first.cell_id in restored_store.cells,
            "newer_frame_not_partially_loaded": second.cell_id not in restored_store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
