from __future__ import annotations

from pathlib import Path

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage127_lagoon_schema_conflict_guard")


class LagoonSchemaConflictGuard:
    """Links contradictory replayed episodes to their matching Lagoon schema without mutation."""

    def detect_and_mark(self, store: OrionMemoryStore, episode: EpisodeFrame) -> list[str]:
        conflicted: list[str] = []
        episode_context = (episode.scene, list(episode.entities), episode.goal, episode.action)
        for semantic in store.cells.values():
            if semantic.memory_system != MemorySystem.SEMANTIC or semantic.source != "lagoon_stage124":
                continue
            signature = semantic.metadata.get("lagoon_signature")
            existing_outcome = semantic.metadata.get("lagoon_schema_outcome")
            if not isinstance(signature, list) or len(signature) != 5 or not isinstance(existing_outcome, str):
                continue
            if tuple(signature[:4]) != episode_context or episode.outcome == existing_outcome:
                continue
            if self._conflict_exists(store, episode.anchor_cell_id, semantic.cell_id):
                conflicted.append(semantic.cell_id)
                continue
            store.mark_conflict(episode.anchor_cell_id, semantic.cell_id, reason="lagoon_outcome_conflict")
            conflicted.append(semantic.cell_id)
        return conflicted

    @staticmethod
    def _conflict_exists(store: OrionMemoryStore, source_cell_id: str, target_cell_id: str) -> bool:
        return any(
            link.source_cell_id == source_cell_id
            and link.target_cell_id == target_cell_id
            and link.link_type == MemoryLinkType.CONFLICT
            for link in store.links
        )


def _bind_replayed_episode(store: OrionMemoryStore, binder: TrifidEpisodeBinder, *, stamp: float, outcome: str, scene: str = "laboratory") -> EpisodeFrame:
    source = store.write_cell(memory_system=MemorySystem.EPISODIC, content=f"robot {scene} sample", summary="sample", source="obs", time_index=stamp)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene=scene, entities=["robot"], goal="collect sample", action="inspect", outcome=outcome)
    completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query=f"robot {scene}", time_index=stamp, window_seconds=1.0)
    assert completion is not None
    TrifidEpisodeReplayer.record_completion(store, completion)
    return frame


def run_stage127_lagoon_conflict_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    first = _bind_replayed_episode(store, binder, stamp=100.0, outcome="secured")
    second = _bind_replayed_episode(store, binder, stamp=130.0, outcome="secured")
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage127")
    semantic_before = store.cells[semantic.semantic_cell_id].to_dict()

    contradictory = _bind_replayed_episode(store, binder, stamp=200.0, outcome="failed")
    guard = LagoonSchemaConflictGuard()
    first_mark = guard.detect_and_mark(store, contradictory)
    link_count = len(store.links)
    second_mark = guard.detect_and_mark(store, contradictory)
    idempotent = second_mark == [semantic.semantic_cell_id] and len(store.links) == link_count
    unrelated = _bind_replayed_episode(store, binder, stamp=230.0, outcome="failed", scene="clinic")
    unrelated_mark = guard.detect_and_mark(store, unrelated)
    conflict_links = [link for link in store.links if link.link_type == MemoryLinkType.CONFLICT]
    summary = {
        "stage": "stage127_lagoon_schema_conflict_guard",
        "semantic_cell_id": semantic.semantic_cell_id,
        "stage_gates": {
            "same_context_opposite_outcome_linked": first_mark == [semantic.semantic_cell_id],
            "repeat_is_idempotent": idempotent,
            "different_context_not_conflicted": unrelated_mark == [],
            "conflict_link_preserves_cells": len(conflict_links) == 1 and conflict_links[0].metadata.get("preserves_original_cells") is True,
            "semantic_cell_unchanged": store.cells[semantic.semantic_cell_id].to_dict() == semantic_before,
            "episodic_cells_preserved": all(frame.anchor_cell_id in store.cells for frame in (first, second, contradictory, unrelated)),
            "conflict_traced": any(event.action.value == "conflict" for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
