from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import TrifidReplayScheduler
from .stage107_trifid_consolidation_candidates import TrifidConsolidationCandidateBuilder
from .stage108_trifid_approved_consolidation import TrifidApprovedConsolidator
from .stage110_trifid_semantic_conflict_guard import TrifidSemanticConflictGuard
from .stage111_trifid_conflict_review import TrifidConflictReview, TrifidConflictReviewBuilder
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage112_trifid_conflict_decision")


@dataclass(frozen=True)
class TrifidConflictDecision:
    decision_cell_id: str
    fingerprint: str
    created: bool
    resolution: str
    approval_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidConflictDecisionLedger:
    """Records an approved preserve-both decision without mutating either fact."""

    def decide(
        self,
        store: OrionMemoryStore,
        review: TrifidConflictReview,
        *,
        approval_id: str,
        resolution: str = "preserve_both",
    ) -> TrifidConflictDecision:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty")
        if resolution != "preserve_both":
            raise ValueError("unsupported resolution")
        fingerprint = f"trifid-conflict:{review.conflict_anchor_cell_id}|{review.semantic_cell_id}|{resolution}"
        existing = next(
            (
                cell
                for cell in store.cells.values()
                if cell.memory_system == MemorySystem.PROCEDURAL
                and cell.metadata.get("trifid_conflict_fingerprint") == fingerprint
            ),
            None,
        )
        if existing is not None:
            return TrifidConflictDecision(existing.cell_id, fingerprint, False, resolution, approval_id)
        decision = store.write_procedural_from_task_trace(
            task_name="trifid_conflict_resolution",
            steps=[
                f"review semantic outcome {review.semantic_outcome}",
                f"review episodic outcome {review.conflict_outcome}",
                "preserve both conflicting memories",
            ],
            outcome="success",
            source="trifid_stage112",
            metadata={
                "trifid_conflict_fingerprint": fingerprint,
                "trifid_resolution": resolution,
                "trifid_approval_id": approval_id,
                "trifid_conflict_episode_id": review.conflict_episode_id,
                "trifid_semantic_cell_id": review.semantic_cell_id,
            },
        )
        for cell_id in (review.conflict_anchor_cell_id, review.semantic_cell_id):
            store.link_cells(
                cell_id,
                decision.cell_id,
                link_type=MemoryLinkType.TASK,
                metadata={"trifid_conflict_resolution": resolution, "fingerprint": fingerprint},
            )
        return TrifidConflictDecision(decision.cell_id, fingerprint, True, resolution, approval_id)


def run_stage112_trifid_conflict_decision_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    semantic = TrifidApprovedConsolidator().consolidate(store, candidate, approval_id="stage112-smoke-approval")
    failed_source = store.write_cell(memory_system="episodic", content="robot failed laboratory collection", summary="failure", source="observation", time_index=400.0)
    failed = binder.bind(store, source_cell_ids=[failed_source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="failed")
    TrifidSemanticConflictGuard().detect_and_mark(store, failed)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    review = TrifidConflictReviewBuilder().build(store, binder, conflict_link)
    assert review is not None
    ledger = TrifidConflictDecisionLedger()
    approval_rejected = unsupported_rejected = False
    try:
        ledger.decide(store, review, approval_id="")
    except ValueError:
        approval_rejected = True
    try:
        ledger.decide(store, review, approval_id="stage112", resolution="prefer_new_episode")
    except ValueError:
        unsupported_rejected = True
    first = ledger.decide(store, review, approval_id="stage112-smoke-approval")
    second = ledger.decide(store, review, approval_id="stage112-repeat-approval")
    task_links = [link for link in store.links if link.link_type == MemoryLinkType.TASK]
    conflict_links = [link for link in store.links if link.link_type == MemoryLinkType.CONFLICT]
    decision = store.cells[first.decision_cell_id]
    summary = {
        "stage": "stage112_trifid_conflict_decision",
        "decision": first.to_dict(),
        "stage_gates": {
            "approval_required": approval_rejected,
            "unsupported_resolution_rejected": unsupported_rejected,
            "procedural_decision_created": first.created and decision.memory_system == MemorySystem.PROCEDURAL,
            "preserve_both_recorded": decision.metadata.get("trifid_resolution") == "preserve_both",
            "both_memories_linked": len(task_links) == 2 and {link.source_cell_id for link in task_links} == {review.conflict_anchor_cell_id, review.semantic_cell_id},
            "repeat_is_idempotent": not second.created and second.decision_cell_id == first.decision_cell_id,
            "conflict_preserved": len(conflict_links) == 1 and semantic.semantic_cell_id in store.cells,
            "source_cells_preserved": all(cell_id in store.cells for cell_id in [*source_ids, failed_source.cell_id]),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
