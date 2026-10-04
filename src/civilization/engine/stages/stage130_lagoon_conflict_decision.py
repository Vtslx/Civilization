from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage127_lagoon_schema_conflict_guard import LagoonSchemaConflictGuard
from .stage129_lagoon_conflict_review import LagoonConflictReview, LagoonConflictReviewBuilder, _bind_replayed_episode
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage130_lagoon_conflict_decision")


@dataclass(frozen=True)
class LagoonConflictDecision:
    decision_cell_id: str
    fingerprint: str
    created: bool
    resolution: str
    approval_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonConflictDecisionLedger:
    """Records an approved preserve-both decision without altering conflicting evidence."""

    def decide(
        self,
        store: OrionMemoryStore,
        review: LagoonConflictReview,
        *,
        approval_id: str,
        resolution: str = "preserve_both",
    ) -> LagoonConflictDecision:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty")
        if resolution != "preserve_both":
            raise ValueError("unsupported resolution")
        if not self._review_matches_store(store, review):
            raise ValueError("review does not match an active Lagoon conflict")
        fingerprint = f"lagoon-conflict:{review.conflict_anchor_cell_id}|{review.semantic_cell_id}|{resolution}"
        existing = next((
            cell for cell in store.cells.values()
            if cell.memory_system == MemorySystem.PROCEDURAL and cell.metadata.get("lagoon_conflict_fingerprint") == fingerprint
        ), None)
        if existing is not None:
            return LagoonConflictDecision(existing.cell_id, fingerprint, False, resolution, approval_id)
        decision = store.write_procedural_from_task_trace(
            task_name="lagoon_conflict_resolution",
            steps=[
                f"review schema outcome {review.schema_outcome}",
                f"review episode outcome {review.conflict_outcome}",
                "preserve both conflicting memories",
            ],
            outcome="success",
            source="lagoon_stage130",
            metadata={
                "lagoon_conflict_fingerprint": fingerprint,
                "lagoon_resolution": resolution,
                "lagoon_approval_id": approval_id,
                "lagoon_conflict_episode_id": review.conflict_episode_id,
                "lagoon_semantic_cell_id": review.semantic_cell_id,
            },
        )
        for cell_id in (review.conflict_anchor_cell_id, review.semantic_cell_id):
            store.link_cells(cell_id, decision.cell_id, link_type=MemoryLinkType.TASK, metadata={"lagoon_conflict_resolution": resolution, "fingerprint": fingerprint})
        return LagoonConflictDecision(decision.cell_id, fingerprint, True, resolution, approval_id)

    @staticmethod
    def _review_matches_store(store: OrionMemoryStore, review: LagoonConflictReview) -> bool:
        semantic = store.cells.get(review.semantic_cell_id)
        if semantic is None or semantic.memory_system != MemorySystem.SEMANTIC or semantic.source != "lagoon_stage124":
            return False
        return any(
            link.link_type == MemoryLinkType.CONFLICT
            and link.source_cell_id == review.conflict_anchor_cell_id
            and link.target_cell_id == review.semantic_cell_id
            and link.metadata.get("reason") == "lagoon_outcome_conflict"
            for link in store.links
        )


def run_stage130_lagoon_conflict_decision_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _bind_replayed_episode(store, binder, stamp=100.0, outcome="secured")
    _bind_replayed_episode(store, binder, stamp=130.0, outcome="secured")
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage130")
    contradictory = _bind_replayed_episode(store, binder, stamp=200.0, outcome="failed")
    LagoonSchemaConflictGuard().detect_and_mark(store, contradictory)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    review = LagoonConflictReviewBuilder().build(store, binder, conflict_link)
    assert review is not None
    semantic_before, conflict_before = store.cells[semantic.semantic_cell_id].to_dict(), conflict_link.to_dict()
    ledger = LagoonConflictDecisionLedger()
    approval_rejected = unsupported_rejected = False
    try:
        ledger.decide(store, review, approval_id="")
    except ValueError:
        approval_rejected = True
    try:
        ledger.decide(store, review, approval_id="stage130", resolution="prefer_new_episode")
    except ValueError:
        unsupported_rejected = True
    first = ledger.decide(store, review, approval_id="stage130-approved")
    second = ledger.decide(store, review, approval_id="stage130-repeat")
    decision = store.cells[first.decision_cell_id]
    task_links = [link for link in store.links if link.link_type == MemoryLinkType.TASK and link.target_cell_id == decision.cell_id]
    summary = {
        "stage": "stage130_lagoon_conflict_decision",
        "decision": first.to_dict(),
        "stage_gates": {
            "approval_required": approval_rejected,
            "unsupported_resolution_rejected": unsupported_rejected,
            "procedural_decision_created": first.created and decision.memory_system == MemorySystem.PROCEDURAL,
            "preserve_both_recorded": decision.metadata.get("lagoon_resolution") == "preserve_both",
            "both_conflicting_memories_linked": {link.source_cell_id for link in task_links} == {review.conflict_anchor_cell_id, review.semantic_cell_id},
            "repeat_is_idempotent": not second.created and second.decision_cell_id == first.decision_cell_id,
            "conflict_evidence_unchanged": store.cells[semantic.semantic_cell_id].to_dict() == semantic_before and conflict_link.to_dict() == conflict_before,
            "conflict_preserved": any(link.link_type == MemoryLinkType.CONFLICT for link in store.links),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
