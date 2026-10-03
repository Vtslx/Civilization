from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import (
    EpisodeCompletion,
    TrifidEpisodePatternCompleter,
)
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage105_trifid_episode_replay")


@dataclass(frozen=True)
class EpisodeReplay:
    episode_id: str
    anchor_cell_id: str
    source_cell_ids: tuple[str, ...]
    created_link_count: int
    already_replayed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidEpisodeReplayer:
    """Records a completed episode as an idempotent, ordered replay sequence."""

    def __init__(self) -> None:
        self._completer = TrifidEpisodePatternCompleter()

    def replay(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        time_index: float,
        window_seconds: float,
    ) -> EpisodeReplay | None:
        completion = self._completer.complete(
            store,
            binder,
            query,
            time_index=time_index,
            window_seconds=window_seconds,
        )
        if completion is None:
            return None
        return self.record_completion(store, completion)

    @staticmethod
    def record_completion(store: OrionMemoryStore, completion: EpisodeCompletion) -> EpisodeReplay:
        return TrifidEpisodeReplayer._record_replay(store, completion)

    @staticmethod
    def _record_replay(store: OrionMemoryStore, completion: EpisodeCompletion) -> EpisodeReplay:
        episode = completion.episode
        source_ids = tuple(cell.cell_id for cell in completion.source_cells)
        existing = {
            (link.target_cell_id, link.metadata.get("replay_step"))
            for link in store.links
            if link.source_cell_id == completion.anchor_cell.cell_id
            and link.link_type == MemoryLinkType.REPLAY
            and link.metadata.get("trifid_episode_id") == episode.episode_id
        }
        created = 0
        for replay_step, source_cell_id in enumerate(source_ids):
            if (source_cell_id, replay_step) in existing:
                continue
            store.link_cells(
                completion.anchor_cell.cell_id,
                source_cell_id,
                link_type=MemoryLinkType.REPLAY,
                metadata={
                    "trifid_episode_id": episode.episode_id,
                    "replay_step": replay_step,
                    "replay_direction": "anchor_to_source",
                },
            )
            created += 1
        return EpisodeReplay(
            episode_id=episode.episode_id,
            anchor_cell_id=completion.anchor_cell.cell_id,
            source_cell_ids=source_ids,
            created_link_count=created,
            already_replayed=created == 0,
        )


def run_stage105_trifid_replay_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=100.0)
    collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=102.0, ttl_seconds=60.0)
    episode = binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
    replayer = TrifidEpisodeReplayer()
    first = replayer.replay(store, binder, "robot laboratory", time_index=101.0, window_seconds=5.0)
    replay_link_count = len([link for link in store.links if link.link_type == MemoryLinkType.REPLAY])
    second = replayer.replay(store, binder, "robot laboratory", time_index=101.0, window_seconds=5.0)
    outside_window = replayer.replay(store, binder, "robot laboratory", time_index=150.0, window_seconds=5.0)
    replay_links = [link for link in store.links if link.link_type == MemoryLinkType.REPLAY]
    summary = {
        "stage": "stage105_trifid_episode_replay",
        "replay": first.to_dict() if first else None,
        "stage_gates": {
            "episode_replayed": first is not None and first.episode_id == episode.episode_id,
            "source_sequence_preserved": first is not None and first.source_cell_ids == (arrival.cell_id, collection.cell_id),
            "ordered_replay_links_created": first is not None and first.created_link_count == 2 and [link.metadata.get("replay_step") for link in replay_links] == [0, 1],
            "repeat_is_idempotent": second is not None and second.already_replayed and len([link for link in store.links if link.link_type == MemoryLinkType.REPLAY]) == replay_link_count,
            "outside_window_rejected": outside_window is None,
            "no_automatic_consolidation": all(cell.memory_system != MemorySystem.SEMANTIC for cell in store.cells.values()),
            "source_cells_preserved": arrival.cell_id in store.cells and collection.cell_id in store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
