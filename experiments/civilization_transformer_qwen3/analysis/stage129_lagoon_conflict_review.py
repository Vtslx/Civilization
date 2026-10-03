from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage127_lagoon_schema_conflict_guard import LagoonSchemaConflictGuard
from .stage73_orion_memory_kernel import MemoryLink, MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage129_lagoon_conflict_review")


@dataclass(frozen=True)
class LagoonConflictReview:
    conflict_episode_id: str
    conflict_anchor_cell_id: str
    semantic_cell_id: str
    schema_outcome: str
    conflict_outcome: str
    support_episode_ids: tuple[str, ...]
    support_anchor_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonConflictReviewBuilder:
    """Builds a provenance-verified, read-only review pack for a Lagoon conflict."""

    def build(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, conflict_link: MemoryLink) -> LagoonConflictReview | None:
        if conflict_link.link_type != MemoryLinkType.CONFLICT or conflict_link.metadata.get("reason") != "lagoon_outcome_conflict":
            return None
        semantic = store.cells.get(conflict_link.target_cell_id)
        conflict_episode = next((frame for frame in binder.frames.values() if frame.anchor_cell_id == conflict_link.source_cell_id), None)
        if semantic is None or semantic.memory_system != MemorySystem.SEMANTIC or semantic.source != "lagoon_stage124" or conflict_episode is None:
            return None
        support_episode_ids = semantic.metadata.get("lagoon_support_episode_ids")
        support_anchor_cell_ids = semantic.metadata.get("consolidated_from")
        schema_outcome = semantic.metadata.get("lagoon_schema_outcome")
        if not isinstance(support_episode_ids, list) or not isinstance(support_anchor_cell_ids, list) or not isinstance(schema_outcome, str):
            return None
        expected_anchors = [binder.frames[episode_id].anchor_cell_id for episode_id in support_episode_ids if episode_id in binder.frames]
        if len(expected_anchors) != len(support_episode_ids) or expected_anchors != support_anchor_cell_ids:
            return None
        evidence = store.read_cells(
            [conflict_link.source_cell_id, semantic.cell_id, *support_anchor_cell_ids],
            trace_details={"lagoon_conflict_review": True, "semantic_cell_id": semantic.cell_id, "conflict_episode_id": conflict_episode.episode_id},
        )
        if len(evidence) != 2 + len(support_anchor_cell_ids):
            return None
        return LagoonConflictReview(
            conflict_episode_id=conflict_episode.episode_id,
            conflict_anchor_cell_id=conflict_link.source_cell_id,
            semantic_cell_id=semantic.cell_id,
            schema_outcome=schema_outcome,
            conflict_outcome=conflict_episode.outcome,
            support_episode_ids=tuple(support_episode_ids),
            support_anchor_cell_ids=tuple(support_anchor_cell_ids),
        )


def _bind_replayed_episode(store: OrionMemoryStore, binder: TrifidEpisodeBinder, *, stamp: float, outcome: str):
    source = store.write_cell(memory_system=MemorySystem.EPISODIC, content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome=outcome)
    completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
    assert completion is not None
    TrifidEpisodeReplayer.record_completion(store, completion)
    return frame


def run_stage129_lagoon_conflict_review_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    first = _bind_replayed_episode(store, binder, stamp=100.0, outcome="secured")
    second = _bind_replayed_episode(store, binder, stamp=130.0, outcome="secured")
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage129")
    contradictory = _bind_replayed_episode(store, binder, stamp=200.0, outcome="failed")
    LagoonSchemaConflictGuard().detect_and_mark(store, contradictory)
    conflict_link = next(link for link in store.links if link.link_type == MemoryLinkType.CONFLICT)
    cell_count, link_count = len(store.cells), len(store.links)
    builder = LagoonConflictReviewBuilder()
    review = builder.build(store, binder, conflict_link)
    temporal_link = next(link for link in store.links if link.link_type == MemoryLinkType.TEMPORAL)
    non_conflict = builder.build(store, binder, temporal_link)
    summary = {
        "stage": "stage129_lagoon_conflict_review",
        "review": review.to_dict() if review else None,
        "stage_gates": {
            "review_built": review is not None and review.semantic_cell_id == semantic.semantic_cell_id,
            "opposing_outcomes_preserved": review is not None and review.schema_outcome == "secured" and review.conflict_outcome == "failed",
            "support_provenance_verified": review is not None and review.support_episode_ids == candidate.support_episode_ids and review.support_anchor_cell_ids == cluster.anchor_cell_ids,
            "non_conflict_rejected": non_conflict is None,
            "review_is_read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "review_read_traced": any(event.details.get("lagoon_conflict_review") for event in store.trace_events),
            "cells_preserved": all(frame.anchor_cell_id in store.cells for frame in (first, second, contradictory)),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
