from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage103_trifid_episode_temporal_index import TrifidTemporalEpisodeRetriever
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from .stage91_orion_persistent_global_store import OrionPersistentGlobalStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage114_trifid_episode_snapshot")


class TrifidEpisodeSnapshot:
    """Persists binder frames separately from the Orion store they reference."""

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, binder: TrifidEpisodeBinder) -> None:
        payload = {
            "version": self.VERSION,
            "next_episode": binder._next_episode,  # noqa: SLF001 - snapshot binder sequence state
            "frames": [frame.to_dict() for frame in binder.frames.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self, store: OrionMemoryStore) -> TrifidEpisodeBinder:
        if not self.path.exists():
            return TrifidEpisodeBinder()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != self.VERSION or not isinstance(payload.get("frames"), list):
            raise ValueError("unsupported Trifid episode snapshot")
        binder = TrifidEpisodeBinder()
        max_sequence = 0
        for item in payload["frames"]:
            frame = EpisodeFrame(
                episode_id=str(item["episode_id"]),
                anchor_cell_id=str(item["anchor_cell_id"]),
                source_cell_ids=tuple(item["source_cell_ids"]),
                scene=str(item["scene"]),
                entities=tuple(item["entities"]),
                goal=str(item["goal"]),
                action=str(item["action"]),
                outcome=str(item["outcome"]),
                cues=tuple(item["cues"]),
                started_at=item.get("started_at"),
                ended_at=item.get("ended_at"),
                time_index=item.get("time_index"),
                sequence=item.get("sequence"),
            )
            self._validate_frame(store, frame)
            if frame.episode_id in binder.frames:
                raise ValueError("duplicate episode_id in snapshot")
            binder.frames[frame.episode_id] = frame
            max_sequence = max(max_sequence, frame.sequence or 0)
        saved_next = payload.get("next_episode", 1)
        if not isinstance(saved_next, int) or saved_next < 1:
            raise ValueError("invalid next_episode in snapshot")
        binder._next_episode = max(saved_next, max_sequence + 1)  # noqa: SLF001 - restore binder sequence state
        return binder

    @staticmethod
    def _validate_frame(store: OrionMemoryStore, frame: EpisodeFrame) -> None:
        anchor = store.cells.get(frame.anchor_cell_id)
        if anchor is None or anchor.memory_system != MemorySystem.EPISODIC:
            raise ValueError("snapshot frame has no episodic anchor")
        if not frame.source_cell_ids or any(cell_id not in store.cells for cell_id in frame.source_cell_ids):
            raise ValueError("snapshot frame has missing source cell")


def run_stage114_trifid_episode_snapshot_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    first = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    second = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=102.0, ttl_seconds=60.0)
    original = binder.bind(store, source_cell_ids=[first.cell_id, second.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
    OrionPersistentGlobalStore(output / "orion_store.json").save(store)
    snapshot = TrifidEpisodeSnapshot(output / "trifid_frames.json")
    snapshot.save(binder)
    restored_store = OrionPersistentGlobalStore(output / "orion_store.json").load()
    restored_binder = snapshot.load(restored_store)
    recalled = TrifidTemporalEpisodeRetriever().retrieve(restored_binder, "robot laboratory", time_index=101.0, window_seconds=5.0)
    next_source = restored_store.write_cell(memory_system="episodic", content="robot exited laboratory", summary="exit", source="observation", time_index=104.0)
    next_frame = restored_binder.bind(restored_store, source_cell_ids=[next_source.cell_id], scene="laboratory", entities=["robot"], goal="leave", action="exit", outcome="complete")
    summary = {
        "stage": "stage114_trifid_episode_snapshot",
        "stage_gates": {
            "snapshot_written": snapshot.path.exists(),
            "frames_restored": set(restored_binder.frames) >= {original.episode_id},
            "restored_retrieval_works": [frame.episode_id for frame in recalled] == [original.episode_id],
            "frame_references_valid": all(frame.anchor_cell_id in restored_store.cells and all(cell_id in restored_store.cells for cell_id in frame.source_cell_ids) for frame in restored_binder.frames.values()),
            "next_episode_monotonic": next_frame.episode_id == "episode-000002",
            "orion_store_restored": first.cell_id in restored_store.cells and second.cell_id in restored_store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
