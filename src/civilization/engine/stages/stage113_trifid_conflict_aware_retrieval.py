from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import TrifidReplayScheduler
from .stage107_trifid_consolidation_candidates import TrifidConsolidationCandidateBuilder
from .stage108_trifid_approved_consolidation import TrifidApprovedConsolidator
from .stage109_trifid_semantic_episode_retrieval import (
    TrifidSemanticEpisodeRecall,
    TrifidSemanticEpisodeRetriever,
)
from .stage110_trifid_semantic_conflict_guard import TrifidSemanticConflictGuard
from .stage111_trifid_conflict_review import TrifidConflictReviewBuilder
from .stage112_trifid_conflict_decision import TrifidConflictDecisionLedger
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage113_trifid_conflict_aware_retrieval")


@dataclass(frozen=True)
class TrifidConflictAwareRecall:
    recall: TrifidSemanticEpisodeRecall
    conflict_episode_ids: tuple[str, ...]
    decision_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall.to_dict(),
            "conflict_episode_ids": list(self.conflict_episode_ids),
            "decision_cell_ids": list(self.decision_cell_ids),
        }


class TrifidConflictAwareRetriever:
    """Returns semantic provenance together with approved preserve-both conflicts."""

    def __init__(self) -> None:
        self._semantic_retriever = TrifidSemanticEpisodeRetriever()

    def retrieve(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        limit: int = 5,
    ) -> list[TrifidConflictAwareRecall]:
        recalls = self._semantic_retriever.retrieve(store, binder, query, limit=limit)
        enriched: list[TrifidConflictAwareRecall] = []
        for recall in recalls:
            conflict_episode_ids, decision_ids = self._approved_conflicts(store, binder, recall.semantic_cell.cell_id)
            evidence_ids = [
                binder.frames[episode_id].anchor_cell_id
                for episode_id in conflict_episode_ids
            ] + list(decision_ids)
            if evidence_ids:
                evidence = store.read_cells(
                    evidence_ids,
                    trace_details={
                        "trifid_conflict_aware_retrieval": True,
                        "semantic_cell_id": recall.semantic_cell.cell_id,
                        "query": query,
                    },
                )
                if len(evidence) != len(evidence_ids):
                    continue
            enriched.append(TrifidConflictAwareRecall(recall, conflict_episode_ids, decision_ids))
        return enriched

    @staticmethod
    def _approved_conflicts(
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        semantic_cell_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        conflict_episode_ids: list[str] = []
        decision_ids: list[str] = []
        for conflict in store.links:
            if conflict.link_type != MemoryLinkType.CONFLICT or conflict.target_cell_id != semantic_cell_id or conflict.metadata.get("reason") != "trifid_outcome_conflict":
                continue
            episode = next((frame for frame in binder.frames.values() if frame.anchor_cell_id == conflict.source_cell_id), None)
            if episode is None:
                continue
            for task_link in store.links:
                if task_link.link_type != MemoryLinkType.TASK or task_link.source_cell_id != semantic_cell_id:
                    continue
                decision = store.cells.get(task_link.target_cell_id)
                if decision is None or decision.memory_system != MemorySystem.PROCEDURAL or decision.metadata.get("trifid_resolution") != "preserve_both":
                    continue
                has_anchor_link = any(
                    link.link_type == MemoryLinkType.TASK
                    and link.source_cell_id == conflict.source_cell_id
                    and link.target_cell_id == decision.cell_id
                    for link in store.links
                )
                if has_anchor_link:
                    conflict_episode_ids.append(episode.episode_id)
                    decision_ids.append(decision.cell_id)
        return tuple(sorted(set(conflict_episode_ids))), tuple(sorted(set(decision_ids)))


def run_stage113_trifid_conflict_aware_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    semantic = TrifidApprovedConsolidator().consolidate(store, candidate, approval_id="stage113-smoke-approval")
    failed_source = store.write_cell(memory_system="episodic", content="robot failed laboratory collection", summary="failure", source="observation", time_index=400.0)
    failed = binder.bind(store, source_cell_ids=[failed_source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="failed")
    TrifidSemanticConflictGuard().detect_and_mark(store, failed)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    review = TrifidConflictReviewBuilder().build(store, binder, conflict_link)
    assert review is not None
    decision = TrifidConflictDecisionLedger().decide(store, review, approval_id="stage113-decision-approval")
    cell_count, link_count = len(store.cells), len(store.links)
    retriever = TrifidConflictAwareRetriever()
    recalls = retriever.retrieve(store, binder, "laboratory collect")
    mismatch = retriever.retrieve(store, binder, "clinic")
    conflict_traces = [event for event in store.trace_events if event.details.get("trifid_conflict_aware_retrieval")]
    summary = {
        "stage": "stage113_trifid_conflict_aware_retrieval",
        "recalls": [recall.to_dict() for recall in recalls],
        "stage_gates": {
            "semantic_recalled": len(recalls) == 1 and recalls[0].recall.semantic_cell.cell_id == semantic.semantic_cell_id,
            "approved_conflict_exposed": len(recalls) == 1 and recalls[0].conflict_episode_ids == (failed.episode_id,) and recalls[0].decision_cell_ids == (decision.decision_cell_id,),
            "mismatch_rejected": mismatch == [],
            "conflict_evidence_traced": len(conflict_traces) == 1,
            "read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "source_cells_preserved": all(cell_id in store.cells for cell_id in [*source_ids, failed_source.cell_id]),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
