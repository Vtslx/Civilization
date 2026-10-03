from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage101_trifid_episode_binding")


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", value.lower()))


@dataclass(frozen=True)
class EpisodeFrame:
    episode_id: str
    anchor_cell_id: str
    source_cell_ids: tuple[str, ...]
    scene: str
    entities: tuple[str, ...]
    goal: str
    action: str
    outcome: str
    cues: tuple[str, ...]
    started_at: float | None = None
    ended_at: float | None = None
    time_index: float | None = None
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidEpisodeBinder:
    def __init__(self) -> None:
        self.frames: dict[str, EpisodeFrame] = {}
        self._next_episode = 1

    def bind(self, store: OrionMemoryStore, *, source_cell_ids: list[str], scene: str, entities: list[str], goal: str, action: str, outcome: str, max_source_gap_seconds: float | None = None) -> EpisodeFrame:
        if not source_cell_ids:
            raise ValueError("source_cell_ids must not be empty")
        for cell_id in source_cell_ids:
            if cell_id not in store.cells or store.cells[cell_id].decay_state != "active":
                raise ValueError("binding sources must be active cells")
        if max_source_gap_seconds is not None and max_source_gap_seconds <= 0:
            raise ValueError("max_source_gap_seconds must be positive")
        source_times = sorted(store.cells[cell_id].time_index for cell_id in source_cell_ids)
        if max_source_gap_seconds is not None and any(
            later - earlier > max_source_gap_seconds
            for earlier, later in zip(source_times, source_times[1:])
        ):
            raise ValueError("binding sources exceed max_source_gap_seconds")
        episode_id = f"episode-{self._next_episode:06d}"; self._next_episode += 1
        cues = tuple(sorted(_terms(" ".join([scene, *entities, goal, action, outcome]))))
        started_at, ended_at = source_times[0], source_times[-1]
        time_index = (started_at + ended_at) / 2
        anchor = store.write_cell(memory_system=MemorySystem.EPISODIC, content=" | ".join(store.cells[cell_id].content for cell_id in source_cell_ids), summary=f"Trifid {scene} {goal}", source="trifid_stage101", time_index=time_index, metadata={"episode_id": episode_id, "scene": scene, "entities": list(entities), "goal": goal, "action": action, "outcome": outcome, "cues": list(cues), "source_cell_ids": list(source_cell_ids), "started_at": started_at, "ended_at": ended_at, "time_index": time_index, "sequence": self._next_episode - 1}, consolidation_state="bound")
        for cell_id in source_cell_ids:
            store.link_cells(cell_id, anchor.cell_id, link_type=MemoryLinkType.TEMPORAL, metadata={"trifid_binding": True, "episode_id": episode_id})
        frame = EpisodeFrame(episode_id, anchor.cell_id, tuple(source_cell_ids), scene, tuple(entities), goal, action, outcome, cues, started_at, ended_at, time_index, self._next_episode - 1)
        self.frames[episode_id] = frame
        return frame

    def retrieve(self, query: str, *, limit: int = 5) -> list[EpisodeFrame]:
        query_terms = _terms(query)
        ranked = [(len(query_terms & set(frame.cues)), frame) for frame in self.frames.values()]
        ranked = [item for item in ranked if item[0] > 0]
        ranked.sort(key=lambda item: (-item[0], item[1].episode_id))
        return [frame for _score, frame in ranked[:limit]]


def run_stage101_trifid_binding_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store = OrionMemoryStore(); binder = TrifidEpisodeBinder()
    a = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation")
    b = store.write_cell(memory_system="working", content="collect sample", summary="goal", source="observation", ttl_seconds=60)
    frame = binder.bind(store, source_cell_ids=[a.cell_id, b.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect bench", outcome="sample secured")
    recalled = binder.retrieve("robot sample")
    summary = {**store.summary(), "episode": frame.to_dict(), "recalled": [item.to_dict() for item in recalled], "stage_gates": {"sources_preserved": a.cell_id in store.cells and b.cell_id in store.cells, "anchor_bound": frame.anchor_cell_id in store.cells, "partial_cue_recalls": [item.episode_id for item in recalled] == [frame.episode_id], "binding_links": len(store.links) == 2}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); store.write_artifacts(output_dir, summary=summary); return summary
