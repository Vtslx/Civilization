from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage126_lagoon_provenance_query import LagoonProvenanceQuery
from .stage127_lagoon_schema_conflict_guard import LagoonSchemaConflictGuard
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage128_lagoon_conflict_aware_recall")


@dataclass(frozen=True)
class LagoonConflictAwareRecall:
    semantic_cell_id: str
    status: str
    eligible_for_context: bool
    conflict_episode_ids: tuple[str, ...]


class LagoonConflictAwareRecallPolicy:
    """Applies a read-time policy to verified Lagoon schemas without changing memory state."""

    def retrieve(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        include_conflicted: bool = True,
    ) -> list[LagoonConflictAwareRecall]:
        recalls: list[LagoonConflictAwareRecall] = []
        frame_by_anchor = {frame.anchor_cell_id: frame.episode_id for frame in binder.frames.values()}
        for recall in LagoonProvenanceQuery().retrieve(store, binder, query):
            conflict_episode_ids = tuple(sorted({
                frame_by_anchor[link.source_cell_id]
                for link in store.links
                if link.link_type == MemoryLinkType.CONFLICT
                and link.target_cell_id == recall.semantic_cell_id
                and link.source_cell_id in frame_by_anchor
            }))
            conflicted = bool(conflict_episode_ids)
            if conflicted and not include_conflicted:
                continue
            recalls.append(LagoonConflictAwareRecall(
                semantic_cell_id=recall.semantic_cell_id,
                status="needs_review" if conflicted else "stable",
                eligible_for_context=not conflicted,
                conflict_episode_ids=conflict_episode_ids,
            ))
        if recalls:
            store.read_cells(
                [recall.semantic_cell_id for recall in recalls],
                trace_details={"lagoon_conflict_aware_recall": True, "query": query, "include_conflicted": include_conflicted},
            )
        return recalls


def _bind_replayed_episode(
    store: OrionMemoryStore,
    binder: TrifidEpisodeBinder,
    *,
    stamp: float,
    outcome: str,
) -> str:
    source = store.write_cell(memory_system=MemorySystem.EPISODIC, content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome=outcome)
    completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
    assert completion is not None
    TrifidEpisodeReplayer.record_completion(store, completion)
    return frame.episode_id


def run_stage128_lagoon_conflict_aware_recall_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _bind_replayed_episode(store, binder, stamp=100.0, outcome="secured")
    _bind_replayed_episode(store, binder, stamp=130.0, outcome="secured")
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage128")
    policy = LagoonConflictAwareRecallPolicy()
    stable = policy.retrieve(store, binder, "laboratory collect")
    semantic_before = store.cells[semantic.semantic_cell_id].to_dict()
    contradictory_episode_id = _bind_replayed_episode(store, binder, stamp=200.0, outcome="failed")
    contradictory = binder.frames[contradictory_episode_id]
    LagoonSchemaConflictGuard().detect_and_mark(store, contradictory)
    surfaced = policy.retrieve(store, binder, "laboratory collect")
    strict = policy.retrieve(store, binder, "laboratory collect", include_conflicted=False)
    summary = {
        "stage": "stage128_lagoon_conflict_aware_recall",
        "stage_gates": {
            "stable_schema_is_context_eligible": len(stable) == 1 and stable[0].status == "stable" and stable[0].eligible_for_context,
            "conflict_is_surfaced_for_review": len(surfaced) == 1 and surfaced[0].status == "needs_review" and not surfaced[0].eligible_for_context and surfaced[0].conflict_episode_ids == (contradictory_episode_id,),
            "strict_mode_excludes_conflicted_schema": strict == [],
            "schema_unchanged_by_recall_policy": store.cells[semantic.semantic_cell_id].to_dict() == semantic_before,
            "conflict_link_preserved": any(link.link_type == MemoryLinkType.CONFLICT and link.target_cell_id == semantic.semantic_cell_id for link in store.links),
            "recall_policy_traced": any(event.details.get("lagoon_conflict_aware_recall") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
