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
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage125_lagoon_schema_retrieval")


@dataclass(frozen=True)
class LagoonSchemaRetrieval:
    semantic_cell_id: str
    support_episode_ids: tuple[str, ...]
    support_anchor_cell_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagoonSchemaRetriever:
    """Retrieves approved Lagoon schemas only when their Orion anchor provenance is readable."""

    def retrieve(self, store: OrionMemoryStore, query: str) -> list[LagoonSchemaRetrieval]:
        if not query.strip():
            raise ValueError("query must not be empty")
        recalls: list[LagoonSchemaRetrieval] = []
        for result in store.read(query, memory_system=MemorySystem.SEMANTIC):
            semantic = result.cell
            if not result.matched_terms or semantic.source != "lagoon_stage124":
                continue
            episodes = semantic.metadata.get("lagoon_support_episode_ids")
            anchors = semantic.metadata.get("consolidated_from")
            if not isinstance(episodes, list) or not isinstance(anchors, list) or len(episodes) != len(anchors) or not anchors:
                continue
            evidence = store.read_cells(anchors, trace_details={"lagoon_schema_retrieval": True, "semantic_cell_id": semantic.cell_id, "query": query})
            if len(evidence) != len(anchors) or any(cell.memory_system != MemorySystem.EPISODIC for cell in evidence):
                continue
            recalls.append(LagoonSchemaRetrieval(semantic.cell_id, tuple(episodes), tuple(anchors)))
        return recalls


def run_stage125_lagoon_schema_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    for stamp in (100.0, 130.0):
        source = store.write_cell(memory_system=MemorySystem.EPISODIC, content="robot laboratory sample", summary="sample", source="obs", time_index=stamp)
        frame = binder.bind(store, source_cell_ids=[source.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured")
        completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query="robot laboratory", time_index=stamp, window_seconds=1.0)
        assert completion is not None
        TrifidEpisodeReplayer.record_completion(store, completion)
    cluster = LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0)[0]
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id="stage125")
    semantic_count = sum(cell.memory_system == MemorySystem.SEMANTIC for cell in store.cells.values())
    retriever = LagoonSchemaRetriever()
    recalls = retriever.retrieve(store, "laboratory collect")
    mismatch = retriever.retrieve(store, "clinic")
    summary = {
        "stage": "stage125_lagoon_schema_retrieval",
        "retrievals": [recall.to_dict() for recall in recalls],
        "stage_gates": {
            "approved_schema_available": len(recalls) == 1 and recalls[0].semantic_cell_id == semantic.semantic_cell_id,
            "anchor_provenance_verified": len(recalls) == 1 and recalls[0].support_anchor_cell_ids == cluster.anchor_cell_ids,
            "support_episode_ids_preserved": len(recalls) == 1 and recalls[0].support_episode_ids == candidate.support_episode_ids,
            "mismatched_query_rejected": mismatch == [],
            "no_extra_semantic_write": sum(cell.memory_system == MemorySystem.SEMANTIC for cell in store.cells.values()) == semantic_count,
            "provenance_read_traced": any(event.details.get("lagoon_schema_retrieval") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
