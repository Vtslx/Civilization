from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore

DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage126_lagoon_provenance_query")


@dataclass(frozen=True)
class LagoonSchemaRecall:
    semantic_cell_id: str
    support_episode_ids: tuple[str, ...]
    anchor_cell_ids: tuple[str, ...]
    conflicted: bool


class LagoonProvenanceQuery:
    def retrieve(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, query: str) -> list[LagoonSchemaRecall]:
        if not query.strip():
            raise ValueError("query must not be empty")
        recalls: list[LagoonSchemaRecall] = []
        for result in store.read(query, memory_system=MemorySystem.SEMANTIC):
            cell = result.cell
            if not result.matched_terms or cell.source != "lagoon_stage124":
                continue
            episodes, anchors = cell.metadata.get("lagoon_support_episode_ids"), cell.metadata.get("consolidated_from")
            if not isinstance(episodes, list) or not isinstance(anchors, list) or any(item not in binder.frames for item in episodes):
                continue
            if [binder.frames[item].anchor_cell_id for item in episodes] != anchors:
                continue
            evidence = store.read_cells(anchors, trace_details={"lagoon_provenance_query": True, "semantic_cell_id": cell.cell_id, "query": query})
            if len(evidence) != len(anchors):
                continue
            conflicted = any(link.link_type == MemoryLinkType.CONFLICT and link.target_cell_id == cell.cell_id for link in store.links)
            recalls.append(LagoonSchemaRecall(cell.cell_id, tuple(episodes), tuple(anchors), conflicted))
        return recalls


def run_stage126_lagoon_provenance_query_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    for stamp in (100.0, 130.0):
        source = store.write_cell(memory_system="episodic", content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
        frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
        completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
        assert completion; TrifidEpisodeReplayer.record_completion(store, completion)
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage126")
    recalls = LagoonProvenanceQuery().retrieve(store, binder, "laboratory collect")
    mismatch = LagoonProvenanceQuery().retrieve(store, binder, "clinic")
    summary = {"stage": "stage126_lagoon_provenance_query", "stage_gates": {"schema_recalled": len(recalls) == 1 and recalls[0].semantic_cell_id == semantic.semantic_cell_id, "provenance_verified": len(recalls) == 1 and recalls[0].support_episode_ids == candidate.support_episode_ids and recalls[0].anchor_cell_ids == cluster.anchor_cell_ids, "mismatch_rejected": mismatch == [], "provenance_traced": any(event.details.get("lagoon_provenance_query") for event in store.trace_events)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
