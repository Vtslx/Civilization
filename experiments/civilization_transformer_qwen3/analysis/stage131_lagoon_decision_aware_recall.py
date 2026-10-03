from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage127_lagoon_schema_conflict_guard import LagoonSchemaConflictGuard
from .stage128_lagoon_conflict_aware_recall import LagoonConflictAwareRecallPolicy
from .stage129_lagoon_conflict_review import LagoonConflictReviewBuilder, _bind_replayed_episode
from .stage130_lagoon_conflict_decision import LagoonConflictDecisionLedger
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage131_lagoon_decision_aware_recall")


@dataclass(frozen=True)
class LagoonDecisionAwareRecall:
    semantic_cell_id: str
    status: str
    eligible_for_context: bool
    decision_cell_id: str | None
    conflict_episode_ids: tuple[str, ...]
    context_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonDecisionAwareRecallPolicy:
    """Surfaces approved preserve-both decisions while keeping control cells out of factual context."""

    def retrieve(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        strict: bool = False,
    ) -> list[LagoonDecisionAwareRecall]:
        base_recalls = LagoonConflictAwareRecallPolicy().retrieve(store, binder, query, include_conflicted=True)
        frame_by_anchor = {frame.anchor_cell_id: frame.episode_id for frame in binder.frames.values()}
        results: list[LagoonDecisionAwareRecall] = []
        for base in base_recalls:
            conflict_links = [
                link for link in store.links
                if link.link_type == MemoryLinkType.CONFLICT and link.target_cell_id == base.semantic_cell_id
            ]
            conflict_episode_ids = tuple(sorted({
                frame_by_anchor[link.source_cell_id]
                for link in conflict_links
                if link.source_cell_id in frame_by_anchor
            }))
            if not conflict_episode_ids:
                result = LagoonDecisionAwareRecall(base.semantic_cell_id, "stable", True, None, (), (base.semantic_cell_id,))
            else:
                decision = self._find_preserve_both_decision(store, base.semantic_cell_id, conflict_links)
                if decision is None:
                    result = LagoonDecisionAwareRecall(base.semantic_cell_id, "needs_review", False, None, conflict_episode_ids, (base.semantic_cell_id,))
                else:
                    context_ids = tuple([base.semantic_cell_id, *sorted({link.source_cell_id for link in conflict_links})])
                    result = LagoonDecisionAwareRecall(base.semantic_cell_id, "preserve_both", True, decision.cell_id, conflict_episode_ids, context_ids)
            if strict and not result.eligible_for_context:
                continue
            evidence = store.read_cells(
                list(result.context_cell_ids),
                trace_details={
                    "lagoon_decision_aware_recall": True,
                    "semantic_cell_id": result.semantic_cell_id,
                    "status": result.status,
                    "decision_cell_id": result.decision_cell_id,
                    "strict": strict,
                },
            )
            if len(evidence) == len(result.context_cell_ids):
                results.append(result)
        return results

    @staticmethod
    def _find_preserve_both_decision(store: OrionMemoryStore, semantic_cell_id: str, conflict_links: list[Any]):
        expected_fingerprints = {
            f"lagoon-conflict:{link.source_cell_id}|{semantic_cell_id}|preserve_both"
            for link in conflict_links
        }
        for cell in store.cells.values():
            if cell.memory_system != MemorySystem.PROCEDURAL or cell.source != "lagoon_stage130":
                continue
            if cell.metadata.get("lagoon_resolution") != "preserve_both" or cell.metadata.get("lagoon_conflict_fingerprint") not in expected_fingerprints:
                continue
            task_sources = {
                link.source_cell_id for link in store.links
                if link.link_type == MemoryLinkType.TASK and link.target_cell_id == cell.cell_id and link.metadata.get("lagoon_conflict_resolution") == "preserve_both"
            }
            if semantic_cell_id in task_sources and any(link.source_cell_id in task_sources for link in conflict_links):
                return cell
        return None


def run_stage131_lagoon_decision_aware_recall_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _bind_replayed_episode(store, binder, stamp=100.0, outcome="secured")
    _bind_replayed_episode(store, binder, stamp=130.0, outcome="secured")
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage131")
    contradictory = _bind_replayed_episode(store, binder, stamp=200.0, outcome="failed")
    LagoonSchemaConflictGuard().detect_and_mark(store, contradictory)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    review = LagoonConflictReviewBuilder().build(store, binder, conflict_link)
    assert review is not None
    policy = LagoonDecisionAwareRecallPolicy()
    pending = policy.retrieve(store, binder, "laboratory collect")
    decision = LagoonConflictDecisionLedger().decide(store, review, approval_id="stage131-approved")
    approved = policy.retrieve(store, binder, "laboratory collect")
    strict = policy.retrieve(store, binder, "laboratory collect", strict=True)
    approved_result = approved[0] if approved else None
    summary = {
        "stage": "stage131_lagoon_decision_aware_recall",
        "decision_cell_id": decision.decision_cell_id,
        "pending": [item.to_dict() for item in pending],
        "approved": [item.to_dict() for item in approved],
        "stage_gates": {
            "unapproved_conflict_needs_review": len(pending) == 1 and pending[0].status == "needs_review" and not pending[0].eligible_for_context,
            "approved_conflict_is_preserve_both": approved_result is not None and approved_result.status == "preserve_both" and approved_result.eligible_for_context,
            "both_fact_cells_are_context_evidence": approved_result is not None and approved_result.context_cell_ids == (semantic.semantic_cell_id, contradictory.anchor_cell_id),
            "decision_cell_not_fact_context": approved_result is not None and decision.decision_cell_id not in approved_result.context_cell_ids,
            "strict_mode_allows_approved_pair": len(strict) == 1 and strict[0].status == "preserve_both",
            "decision_link_is_traceable": approved_result is not None and approved_result.decision_cell_id == decision.decision_cell_id,
            "recall_is_traced": any(event.details.get("lagoon_decision_aware_recall") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
