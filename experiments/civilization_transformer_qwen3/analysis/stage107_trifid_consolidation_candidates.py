from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import ScheduledReplay, TrifidReplayScheduler
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage107_trifid_consolidation_candidates")


@dataclass(frozen=True)
class TrifidConsolidationCandidate:
    signature: tuple[str, tuple[str, ...], str, str, str]
    episode_ids: tuple[str, ...]
    anchor_cell_ids: tuple[str, ...]
    source_cell_ids: tuple[str, ...]
    average_confidence: float
    average_importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidConsolidationCandidateBuilder:
    """Produces evidence-only consolidation candidates from complete replayed episodes."""

    def build(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        scheduled: list[ScheduledReplay],
        *,
        min_episodes: int = 2,
    ) -> list[TrifidConsolidationCandidate]:
        if min_episodes < 2:
            raise ValueError("min_episodes must be >= 2")
        episodes = [binder.frames[item.episode_id] for item in scheduled if item.episode_id in binder.frames]
        anchors = store.read_cells(
            [episode.anchor_cell_id for episode in episodes],
            trace_details={
                "trifid_consolidation_candidate": True,
                "scheduled_episode_ids": [episode.episode_id for episode in episodes],
            },
        )
        anchors_by_id = {anchor.cell_id: anchor for anchor in anchors}
        groups: dict[tuple[str, tuple[str, ...], str, str, str], list[EpisodeFrame]] = {}
        for episode in episodes:
            if episode.anchor_cell_id not in anchors_by_id or not self._has_complete_replay(store, episode):
                continue
            groups.setdefault(self._signature(episode), []).append(episode)
        candidates: list[TrifidConsolidationCandidate] = []
        for signature, group in groups.items():
            if len(group) < min_episodes:
                continue
            group.sort(key=lambda episode: (episode.time_index or 0.0, episode.sequence or 0, episode.episode_id))
            anchor_cells = [anchors_by_id[episode.anchor_cell_id] for episode in group]
            candidates.append(
                TrifidConsolidationCandidate(
                    signature=signature,
                    episode_ids=tuple(episode.episode_id for episode in group),
                    anchor_cell_ids=tuple(episode.anchor_cell_id for episode in group),
                    source_cell_ids=tuple(cell_id for episode in group for cell_id in episode.source_cell_ids),
                    average_confidence=sum(anchor.confidence for anchor in anchor_cells) / len(anchor_cells),
                    average_importance=sum(anchor.importance for anchor in anchor_cells) / len(anchor_cells),
                )
            )
        candidates.sort(key=lambda candidate: (-len(candidate.episode_ids), candidate.signature, candidate.episode_ids))
        return candidates

    @staticmethod
    def _signature(episode: EpisodeFrame) -> tuple[str, tuple[str, ...], str, str, str]:
        return (episode.scene, episode.entities, episode.goal, episode.action, episode.outcome)

    @staticmethod
    def _has_complete_replay(store: OrionMemoryStore, episode: EpisodeFrame) -> bool:
        observed_steps = {
            (link.target_cell_id, link.metadata.get("replay_step"))
            for link in store.links
            if link.source_cell_id == episode.anchor_cell_id
            and link.link_type == MemoryLinkType.REPLAY
            and link.metadata.get("trifid_episode_id") == episode.episode_id
        }
        return all((source_cell_id, step) in observed_steps for step, source_cell_id in enumerate(episode.source_cell_ids))


def run_stage107_trifid_consolidation_candidate_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    episodes: list[EpisodeFrame] = []
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        episode = binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        episodes.append(episode)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduler = TrifidReplayScheduler()
    scheduled = scheduler.schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    builder = TrifidConsolidationCandidateBuilder()
    candidates = builder.build(store, binder, scheduled, min_episodes=2)
    insufficient = builder.build(store, binder, scheduled[:1], min_episodes=2)
    candidate_traces = [event for event in store.trace_events if event.details.get("trifid_consolidation_candidate")]
    summary = {
        "stage": "stage107_trifid_consolidation_candidates",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "stage_gates": {
            "replayed_evidence_grouped": len(candidates) == 1 and set(candidates[0].episode_ids) == {item.episode_id for item in scheduled},
            "complete_replay_required": all(len(candidate.source_cell_ids) == 2 * len(candidate.episode_ids) for candidate in candidates),
            "insufficient_evidence_rejected": insufficient == [],
            "candidate_read_traced": len(candidate_traces) == 2,
            "no_semantic_written": all(cell.memory_system != MemorySystem.SEMANTIC for cell in store.cells.values()),
            "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
