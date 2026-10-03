from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage103_trifid_episode_temporal_index import TrifidTemporalEpisodeRetriever
from .stage114_trifid_episode_snapshot import TrifidEpisodeSnapshot
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from .stage91_orion_persistent_global_store import OrionPersistentGlobalStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage116_trifid_paired_checkpoint")


class TrifidPairedCheckpoint:
    """Persists and verifies one compatible Orion-store/Trifid-frame checkpoint."""

    MANIFEST = "trifid_checkpoint_manifest.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.store_path = self.directory / "orion_store.json"
        self.frames_path = self.directory / "trifid_frames.json"
        self.manifest_path = self.directory / self.MANIFEST

    def save(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder) -> None:
        OrionPersistentGlobalStore(self.store_path).save(store)
        TrifidEpisodeSnapshot(self.frames_path).save(binder)
        manifest = {
            "version": 1,
            "files": {
                self.store_path.name: self._sha256(self.store_path),
                self.frames_path.name: self._sha256(self.frames_path),
            },
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def load(self) -> tuple[OrionMemoryStore, TrifidEpisodeBinder]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("unsupported Trifid checkpoint manifest")
        for path in (self.store_path, self.frames_path):
            if manifest["files"].get(path.name) != self._sha256(path):
                raise ValueError("Trifid checkpoint integrity check failed")
        store = OrionPersistentGlobalStore(self.store_path).load()
        return store, TrifidEpisodeSnapshot(self.frames_path).load(store)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage116_trifid_paired_checkpoint_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="observe", action="inspect", outcome="secured")
    checkpoint = TrifidPairedCheckpoint(output / "checkpoint")
    checkpoint.save(store, binder)
    restored_store, restored_binder = checkpoint.load()
    recalled = TrifidTemporalEpisodeRetriever().retrieve(restored_binder, "robot laboratory", time_index=100.0, window_seconds=1.0)
    checkpoint.frames_path.write_text("{}", encoding="utf-8")
    tamper_rejected = False
    try:
        checkpoint.load()
    except ValueError:
        tamper_rejected = True
    summary = {
        "stage": "stage116_trifid_paired_checkpoint",
        "stage_gates": {
            "manifest_written": checkpoint.manifest_path.exists(),
            "paired_state_restored": source.cell_id in restored_store.cells and frame.episode_id in restored_binder.frames,
            "restored_retrieval_works": [item.episode_id for item in recalled] == [frame.episode_id],
            "tamper_rejected_before_load": tamper_rejected,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
