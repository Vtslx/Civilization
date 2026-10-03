from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage116_trifid_paired_checkpoint import TrifidPairedCheckpoint
from .stage117_trifid_checkpoint_recovery import TrifidCheckpointRepository
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage119_trifid_checkpoint_audit")


class TrifidCheckpointAuditor:
    """Builds a read-only integrity report for checkpoint generations."""

    def audit(self, repository: TrifidCheckpointRepository) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted((item for item in repository.directory.glob("checkpoint-*") if item.is_dir()), reverse=True):
            try:
                store, binder = TrifidPairedCheckpoint(path).load()
                entries.append({"generation": path.name, "valid": True, "reason": None, "cell_count": len(store.cells), "frame_count": len(binder.frames)})
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                entries.append({"generation": path.name, "valid": False, "reason": type(error).__name__, "cell_count": None, "frame_count": None})
        return entries


def run_stage119_trifid_checkpoint_audit_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir); repo = TrifidCheckpointRepository(output / "checkpoints")
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source = store.write_cell(memory_system="episodic", content="robot event", summary="event", source="obs", time_index=1.0)
    binder.bind(store, source_cell_ids=[source.cell_id], scene="lab", entities=["robot"], goal="observe", action="inspect", outcome="done")
    valid = repo.save(1, store, binder)
    corrupt = repo.save(2, store, binder); (corrupt / "orion_store.json").write_text("{}", encoding="utf-8")
    before = {str(path): path.stat().st_mtime_ns for path in (valid / "orion_store.json", valid / "trifid_frames.json", corrupt / "orion_store.json", corrupt / "trifid_frames.json")}
    entries = TrifidCheckpointAuditor().audit(repo)
    after = {str(path): path.stat().st_mtime_ns for path in (valid / "orion_store.json", valid / "trifid_frames.json", corrupt / "orion_store.json", corrupt / "trifid_frames.json")}
    summary = {"stage": "stage119_trifid_checkpoint_audit", "entries": entries, "stage_gates": {"valid_generation_reported": any(item["generation"] == valid.name and item["valid"] and item["frame_count"] == 1 for item in entries), "corrupt_generation_reported": any(item["generation"] == corrupt.name and not item["valid"] for item in entries), "audit_read_only": before == after}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
