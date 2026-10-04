from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage116_trifid_paired_checkpoint import TrifidPairedCheckpoint
from .stage117_trifid_checkpoint_recovery import TrifidCheckpointRepository
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage118_trifid_checkpoint_retention")


class TrifidCheckpointRetention:
    """Retains recent verified checkpoints without deleting corrupt forensic evidence."""

    def prune(self, repository: TrifidCheckpointRepository, *, keep_valid: int) -> dict[str, list[str]]:
        if keep_valid < 1:
            raise ValueError("keep_valid must be >= 1")
        paths = sorted((path for path in repository.directory.glob("checkpoint-*") if path.is_dir()), reverse=True)
        valid: list[Path] = []
        corrupt: list[str] = []
        for path in paths:
            try:
                TrifidPairedCheckpoint(path).load()
                valid.append(path)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                corrupt.append(path.name)
        removed: list[str] = []
        for path in valid[keep_valid:]:
            shutil.rmtree(path)
            removed.append(path.name)
        return {"retained": [path.name for path in valid[:keep_valid]], "removed": removed, "corrupt_preserved": corrupt}


def run_stage118_trifid_checkpoint_retention_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir); repo = TrifidCheckpointRepository(output / "checkpoints")
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    generations: list[Path] = []
    for generation in (1, 2, 3):
        source = store.write_cell(memory_system="episodic", content=f"robot event {generation}", summary=f"event {generation}", source="obs", time_index=float(generation))
        binder.bind(store, source_cell_ids=[source.cell_id], scene="lab", entities=["robot"], goal=f"goal {generation}", action="inspect", outcome="done")
        generations.append(repo.save(generation, store, binder))
    (generations[2] / "trifid_frames.json").write_text("{}", encoding="utf-8")
    result = TrifidCheckpointRetention().prune(repo, keep_valid=1)
    summary = {"stage": "stage118_trifid_checkpoint_retention", "retention": result, "stage_gates": {"latest_valid_retained": result["retained"] == [generations[1].name] and generations[1].exists(), "old_valid_removed": result["removed"] == [generations[0].name] and not generations[0].exists(), "corrupt_preserved": result["corrupt_preserved"] == [generations[2].name] and generations[2].exists()}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
