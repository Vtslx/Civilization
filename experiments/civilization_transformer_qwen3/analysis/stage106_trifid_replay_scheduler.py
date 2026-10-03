from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage103_trifid_episode_temporal_index import TrifidTemporalEpisodeRetriever
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import EpisodeReplay, TrifidEpisodeReplayer
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage106_trifid_replay_scheduler")


@dataclass(frozen=True)
class ScheduledReplay:
    episode_id: str
    priority: float
    replay: EpisodeReplay

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["replay"] = self.replay.to_dict()
        return payload


class TrifidReplayScheduler:
    """Selects a bounded set of time-safe episodes for deterministic replay."""

    def __init__(self) -> None:
        self._temporal_retriever = TrifidTemporalEpisodeRetriever()
        self._completer = TrifidEpisodePatternCompleter()
        self._replayer = TrifidEpisodeReplayer()

    def schedule_and_replay(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        time_index: float,
        window_seconds: float,
        budget: int,
    ) -> list[ScheduledReplay]:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        candidates = self._temporal_retriever.retrieve(
            binder,
            query,
            time_index=time_index,
            window_seconds=window_seconds,
            limit=len(binder.frames),
        )
        ranked = self._rank(store, candidates)
        scheduled: list[ScheduledReplay] = []
        for priority, episode in ranked[:budget]:
            completion = self._completer.complete_frame(
                store,
                episode,
                query=query,
                time_index=time_index,
                window_seconds=window_seconds,
            )
            if completion is None:
                continue
            scheduled.append(
                ScheduledReplay(
                    episode_id=episode.episode_id,
                    priority=priority,
                    replay=self._replayer.record_completion(store, completion),
                )
            )
        return scheduled

    @staticmethod
    def _rank(store: OrionMemoryStore, episodes: list[EpisodeFrame]) -> list[tuple[float, EpisodeFrame]]:
        ranked: list[tuple[float, EpisodeFrame]] = []
        for episode in episodes:
            anchor = store.cells.get(episode.anchor_cell_id)
            if anchor is None or anchor.decay_state != "active":
                continue
            priority = 0.7 * anchor.importance + 0.3 * anchor.confidence
            ranked.append((priority, episode))
        ranked.sort(
            key=lambda item: (
                -item[0],
                -(item[1].time_index or 0.0),
                item[1].sequence or 0,
                item[1].episode_id,
            )
        )
        return ranked


def run_stage106_trifid_replay_scheduler_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    episodes: list[EpisodeFrame] = []
    source_ids: list[str] = []
    for start, importance, confidence in ((100.0, 0.9, 0.9), (200.0, 0.8, 1.0), (300.0, 0.6, 0.9)):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        episode = binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        store.update_cell(episode.anchor_cell_id, importance=importance, confidence=confidence)
        episodes.append(episode)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduler = TrifidReplayScheduler()
    first = scheduler.schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    replay_link_count = len([link for link in store.links if link.link_type.value == "replay"])
    second = scheduler.schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    outside_window = scheduler.schedule_and_replay(store, binder, "robot laboratory", time_index=500.0, window_seconds=5.0, budget=2)
    summary = {
        "stage": "stage106_trifid_replay_scheduler",
        "scheduled": [item.to_dict() for item in first],
        "stage_gates": {
            "budget_respected": len(first) == 2,
            "priority_ordered": [item.episode_id for item in first] == [episodes[0].episode_id, episodes[1].episode_id],
            "replay_links_written": replay_link_count == 4,
            "repeat_is_idempotent": len(second) == 2 and all(item.replay.already_replayed for item in second) and len([link for link in store.links if link.link_type.value == "replay"]) == replay_link_count,
            "outside_window_rejected": outside_window == [],
            "no_automatic_consolidation": all(cell.memory_system != MemorySystem.SEMANTIC for cell in store.cells.values()),
            "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
