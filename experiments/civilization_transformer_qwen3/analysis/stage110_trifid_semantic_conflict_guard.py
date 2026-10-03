from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import TrifidReplayScheduler
from .stage107_trifid_consolidation_candidates import TrifidConsolidationCandidateBuilder
from .stage108_trifid_approved_consolidation import TrifidApprovedConsolidator
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage110_trifid_semantic_conflict_guard")


@dataclass(frozen=True)
class TrifidSemanticConflict:
    episode_id: str
    semantic_cell_id: str
    created: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidSemanticConflictGuard:
    """Separates a conflicting new episode from a consolidated semantic pattern."""

    def detect_and_mark(
        self,
        store: OrionMemoryStore,
        episode: EpisodeFrame,
    ) -> list[TrifidSemanticConflict]:
        conflicts: list[TrifidSemanticConflict] = []
        for semantic in store.cells.values():
            if semantic.memory_system != MemorySystem.SEMANTIC:
                continue
            semantic_signature = self._signature(semantic.metadata.get("trifid_candidate_signature"))
            if semantic_signature is None or not self._has_conflicting_outcome(episode, semantic_signature):
                continue
            existing = any(
                link.source_cell_id == episode.anchor_cell_id
                and link.target_cell_id == semantic.cell_id
                and link.link_type == MemoryLinkType.CONFLICT
                and link.metadata.get("reason") == "trifid_outcome_conflict"
                for link in store.links
            )
            if not existing:
                store.mark_conflict(
                    episode.anchor_cell_id,
                    semantic.cell_id,
                    reason="trifid_outcome_conflict",
                )
            conflicts.append(TrifidSemanticConflict(episode.episode_id, semantic.cell_id, not existing))
        return conflicts

    @staticmethod
    def _signature(value: Any) -> tuple[str, tuple[str, ...], str, str, str] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 5 or not isinstance(value[1], (list, tuple)):
            return None
        return (str(value[0]), tuple(str(entity) for entity in value[1]), str(value[2]), str(value[3]), str(value[4]))

    @staticmethod
    def _has_conflicting_outcome(
        episode: EpisodeFrame,
        semantic_signature: tuple[str, tuple[str, ...], str, str, str],
    ) -> bool:
        episode_context = (episode.scene, episode.entities, episode.goal, episode.action)
        semantic_context = semantic_signature[:4]
        return episode_context == semantic_context and episode.outcome != semantic_signature[4]


def run_stage110_trifid_semantic_conflict_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    semantic = TrifidApprovedConsolidator().consolidate(store, candidate, approval_id="stage110-smoke-approval")
    failed_source = store.write_cell(memory_system="episodic", content="robot failed laboratory collection", summary="failure", source="observation", time_index=400.0)
    failed = binder.bind(store, source_cell_ids=[failed_source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="failed")
    matching_source = store.write_cell(memory_system="episodic", content="robot secured laboratory collection", summary="success", source="observation", time_index=500.0)
    matching = binder.bind(store, source_cell_ids=[matching_source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
    guard = TrifidSemanticConflictGuard()
    first = guard.detect_and_mark(store, failed)
    second = guard.detect_and_mark(store, failed)
    no_conflict = guard.detect_and_mark(store, matching)
    conflict_links = [link for link in store.links if link.link_type == MemoryLinkType.CONFLICT]
    conflict_traces = [event for event in store.trace_events if event.action.value == "conflict"]
    summary = {
        "stage": "stage110_trifid_semantic_conflict_guard",
        "conflicts": [conflict.to_dict() for conflict in first],
        "stage_gates": {
            "outcome_conflict_marked": len(first) == 1 and first[0].semantic_cell_id == semantic.semantic_cell_id and first[0].created,
            "conflict_link_preserves_cells": len(conflict_links) == 1 and conflict_links[0].metadata.get("preserves_original_cells") is True,
            "repeat_is_idempotent": len(second) == 1 and not second[0].created and len(conflict_links) == 1,
            "matching_outcome_not_conflicted": no_conflict == [],
            "conflict_traced": len(conflict_traces) == 1,
            "semantic_unchanged": sum(cell.memory_system == MemorySystem.SEMANTIC for cell in store.cells.values()) == 1,
            "source_cells_preserved": all(cell_id in store.cells for cell_id in [*source_ids, failed_source.cell_id, matching_source.cell_id]),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
