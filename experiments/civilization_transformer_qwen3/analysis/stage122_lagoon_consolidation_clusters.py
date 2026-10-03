from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage122_lagoon_consolidation_clusters")


@dataclass(frozen=True)
class LagoonConsolidationCluster:
    signature: tuple[str, tuple[str, ...], str, str, str]
    episode_ids: tuple[str, ...]
    anchor_cell_ids: tuple[str, ...]
    started_at: float
    ended_at: float
    average_confidence: float
    average_importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonConsolidationClusterer:
    """Builds time-continuous, replay-backed consolidation evidence without writing semantic memory."""

    def cluster(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, *, min_episodes: int = 2, max_gap_seconds: float = 300.0) -> list[LagoonConsolidationCluster]:
        if min_episodes < 2 or max_gap_seconds <= 0:
            raise ValueError("min_episodes must be >= 2 and max_gap_seconds must be positive")
        groups: dict[tuple[str, tuple[str, ...], str, str, str], list[EpisodeFrame]] = {}
        for frame in binder.frames.values():
            if self._replayed(store, frame):
                groups.setdefault(self._signature(frame), []).append(frame)
        clusters: list[LagoonConsolidationCluster] = []
        for signature, frames in groups.items():
            frames.sort(key=lambda frame: (frame.time_index or 0.0, frame.sequence or 0, frame.episode_id))
            window: list[EpisodeFrame] = []
            for frame in frames:
                if window and (frame.time_index or 0.0) - (window[-1].time_index or 0.0) > max_gap_seconds:
                    clusters.extend(self._emit(store, signature, window, min_episodes))
                    window = []
                window.append(frame)
            clusters.extend(self._emit(store, signature, window, min_episodes))
        clusters.sort(key=lambda cluster: (cluster.started_at, cluster.episode_ids))
        if clusters:
            store.read_cells([anchor for cluster in clusters for anchor in cluster.anchor_cell_ids], trace_details={"lagoon_consolidation_cluster": True})
        return clusters

    @staticmethod
    def _emit(store: OrionMemoryStore, signature: tuple[str, tuple[str, ...], str, str, str], frames: list[EpisodeFrame], min_episodes: int) -> list[LagoonConsolidationCluster]:
        if len(frames) < min_episodes:
            return []
        anchors = [store.cells[frame.anchor_cell_id] for frame in frames]
        return [LagoonConsolidationCluster(signature, tuple(frame.episode_id for frame in frames), tuple(frame.anchor_cell_id for frame in frames), frames[0].started_at or 0.0, frames[-1].ended_at or 0.0, sum(anchor.confidence for anchor in anchors) / len(anchors), sum(anchor.importance for anchor in anchors) / len(anchors))]

    @staticmethod
    def _signature(frame: EpisodeFrame) -> tuple[str, tuple[str, ...], str, str, str]:
        return (frame.scene, frame.entities, frame.goal, frame.action, frame.outcome)

    @staticmethod
    def _replayed(store: OrionMemoryStore, frame: EpisodeFrame) -> bool:
        steps = {(link.target_cell_id, link.metadata.get("replay_step")) for link in store.links if link.source_cell_id == frame.anchor_cell_id and link.link_type == MemoryLinkType.REPLAY and link.metadata.get("trifid_episode_id") == frame.episode_id}
        return all((cell_id, step) in steps for step, cell_id in enumerate(frame.source_cell_ids))


def run_stage122_lagoon_cluster_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder(); frames: list[EpisodeFrame] = []; source_ids: list[str] = []
    for stamp in (100.0, 130.0, 1000.0):
        source = store.write_cell(memory_system="episodic", content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
        frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
        completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
        assert completion is not None; TrifidEpisodeReplayer.record_completion(store, completion); frames.append(frame); source_ids.append(source.cell_id)
    clusters = LagoonConsolidationClusterer().cluster(store, binder, min_episodes=2, max_gap_seconds=50.0)
    summary = {"stage": "stage122_lagoon_consolidation_clusters", "clusters": [cluster.to_dict() for cluster in clusters], "stage_gates": {"continuous_replayed_episodes_clustered": len(clusters) == 1 and clusters[0].episode_ids == (frames[0].episode_id, frames[1].episode_id), "late_episode_not_forced_into_cluster": frames[2].episode_id not in (clusters[0].episode_ids if clusters else ()), "no_semantic_written": all(cell.memory_system != MemorySystem.SEMANTIC for cell in store.cells.values()), "cluster_read_traced": any(event.details.get("lagoon_consolidation_cluster") for event in store.trace_events), "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values()); store.write_artifacts(output_dir, summary=summary); return summary
