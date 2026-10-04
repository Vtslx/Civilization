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
from .stage73_orion_memory_kernel import MemoryLink, MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage111_trifid_conflict_review")


@dataclass(frozen=True)
class TrifidConflictReview:
    conflict_episode_id: str
    conflict_anchor_cell_id: str
    semantic_cell_id: str
    semantic_outcome: str
    conflict_outcome: str
    prior_episode_ids: tuple[str, ...]
    prior_anchor_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidConflictReviewBuilder:
    """Builds a read-only review pack from an explicit Trifid outcome conflict."""

    def build(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        conflict_link: MemoryLink,
    ) -> TrifidConflictReview | None:
        if conflict_link.link_type != MemoryLinkType.CONFLICT or conflict_link.metadata.get("reason") != "trifid_outcome_conflict":
            return None
        semantic = store.cells.get(conflict_link.target_cell_id)
        if semantic is None or semantic.memory_system != MemorySystem.SEMANTIC:
            return None
        conflict_episode = next(
            (frame for frame in binder.frames.values() if frame.anchor_cell_id == conflict_link.source_cell_id),
            None,
        )
        anchor_ids = semantic.metadata.get("consolidated_from")
        prior_episode_ids = semantic.metadata.get("trifid_candidate_episode_ids")
        signature = semantic.metadata.get("trifid_candidate_signature")
        if conflict_episode is None or not isinstance(anchor_ids, list) or not isinstance(prior_episode_ids, list) or not isinstance(signature, (list, tuple)) or len(signature) != 5:
            return None
        expected_anchor_ids = [binder.frames[episode_id].anchor_cell_id for episode_id in prior_episode_ids if episode_id in binder.frames]
        if len(expected_anchor_ids) != len(prior_episode_ids) or expected_anchor_ids != anchor_ids:
            return None
        evidence = store.read_cells(
            [conflict_link.source_cell_id, conflict_link.target_cell_id, *anchor_ids],
            trace_details={
                "trifid_conflict_review": True,
                "conflict_episode_id": conflict_episode.episode_id,
                "semantic_cell_id": semantic.cell_id,
            },
        )
        if len(evidence) != 2 + len(anchor_ids):
            return None
        return TrifidConflictReview(
            conflict_episode_id=conflict_episode.episode_id,
            conflict_anchor_cell_id=conflict_link.source_cell_id,
            semantic_cell_id=semantic.cell_id,
            semantic_outcome=str(signature[4]),
            conflict_outcome=conflict_episode.outcome,
            prior_episode_ids=tuple(prior_episode_ids),
            prior_anchor_cell_ids=tuple(anchor_ids),
        )


def run_stage111_trifid_conflict_review_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    semantic = TrifidApprovedConsolidator().consolidate(store, candidate, approval_id="stage111-smoke-approval")
    failed_source = store.write_cell(memory_system="episodic", content="robot failed laboratory collection", summary="failure", source="observation", time_index=400.0)
    failed = binder.bind(store, source_cell_ids=[failed_source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="failed")
    TrifidSemanticConflictGuard().detect_and_mark(store, failed)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    cell_count, link_count = len(store.cells), len(store.links)
    builder = TrifidConflictReviewBuilder()
    review = builder.build(store, binder, conflict_link)
    non_conflict = builder.build(store, binder, next(link for link in store.links if link.link_type == MemoryLinkType.TEMPORAL))
    review_traces = [event for event in store.trace_events if event.details.get("trifid_conflict_review")]
    summary = {
        "stage": "stage111_trifid_conflict_review",
        "review": review.to_dict() if review else None,
        "stage_gates": {
            "review_built": review is not None and review.semantic_cell_id == semantic.semantic_cell_id,
            "both_outcomes_preserved": review is not None and review.semantic_outcome == "secured" and review.conflict_outcome == "failed",
            "prior_provenance_preserved": review is not None and review.prior_episode_ids == tuple(candidate.episode_ids) and review.prior_anchor_cell_ids == tuple(candidate.anchor_cell_ids),
            "non_conflict_rejected": non_conflict is None,
            "review_read_traced": len(review_traces) == 1,
            "read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "source_cells_preserved": all(cell_id in store.cells for cell_id in [*source_ids, failed_source.cell_id]),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
