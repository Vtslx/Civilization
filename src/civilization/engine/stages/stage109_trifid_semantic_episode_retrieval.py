from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import TrifidReplayScheduler
from .stage107_trifid_consolidation_candidates import TrifidConsolidationCandidateBuilder
from .stage108_trifid_approved_consolidation import TrifidApprovedConsolidator
from .stage73_orion_memory_kernel import MemoryCell, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage109_trifid_semantic_episode_retrieval")


@dataclass(frozen=True)
class TrifidSemanticEpisodeRecall:
    semantic_cell: MemoryCell
    episode_ids: tuple[str, ...]
    anchor_cells: tuple[MemoryCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_cell": self.semantic_cell.to_dict(),
            "episode_ids": list(self.episode_ids),
            "anchor_cells": [cell.to_dict() for cell in self.anchor_cells],
        }


class TrifidSemanticEpisodeRetriever:
    """Follows explicit semantic provenance back to consolidated episodic anchors."""

    def retrieve(
        self,
        store: OrionMemoryStore,
        binder: TrifidEpisodeBinder,
        query: str,
        *,
        limit: int = 5,
    ) -> list[TrifidSemanticEpisodeRecall]:
        if not query.strip():
            raise ValueError("query must not be empty")
        semantic_results = store.read(query, memory_system=MemorySystem.SEMANTIC, limit=limit)
        recalls: list[TrifidSemanticEpisodeRecall] = []
        for result in semantic_results:
            if not result.matched_terms:
                continue
            anchor_ids = result.cell.metadata.get("consolidated_from")
            episode_ids = result.cell.metadata.get("trifid_candidate_episode_ids")
            if not isinstance(anchor_ids, list) or not isinstance(episode_ids, list):
                continue
            anchors = store.read_cells(
                anchor_ids,
                trace_details={
                    "trifid_semantic_provenance": True,
                    "semantic_cell_id": result.cell.cell_id,
                    "query": query,
                },
            )
            if len(anchors) != len(anchor_ids) or any(anchor.memory_system != MemorySystem.EPISODIC for anchor in anchors):
                continue
            if any(episode_id not in binder.frames for episode_id in episode_ids):
                continue
            expected_anchor_ids = [binder.frames[episode_id].anchor_cell_id for episode_id in episode_ids]
            if expected_anchor_ids != anchor_ids:
                continue
            recalls.append(
                TrifidSemanticEpisodeRecall(
                    semantic_cell=result.cell,
                    episode_ids=tuple(episode_ids),
                    anchor_cells=tuple(anchors),
                )
            )
        return recalls


def run_stage109_trifid_semantic_episode_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    consolidation = TrifidApprovedConsolidator().consolidate(store, candidate, approval_id="stage109-smoke-approval")
    retriever = TrifidSemanticEpisodeRetriever()
    recalls = retriever.retrieve(store, binder, "laboratory collect")
    mismatch = retriever.retrieve(store, binder, "clinic")
    provenance_traces = [event for event in store.trace_events if event.details.get("trifid_semantic_provenance")]
    semantic_count = sum(cell.memory_system == MemorySystem.SEMANTIC for cell in store.cells.values())
    summary = {
        "stage": "stage109_trifid_semantic_episode_retrieval",
        "recalls": [recall.to_dict() for recall in recalls],
        "stage_gates": {
            "semantic_recalled": len(recalls) == 1 and recalls[0].semantic_cell.cell_id == consolidation.semantic_cell_id,
            "provenance_only": len(recalls) == 1 and recalls[0].episode_ids == tuple(candidate.episode_ids) and [anchor.cell_id for anchor in recalls[0].anchor_cells] == list(candidate.anchor_cell_ids),
            "mismatched_query_rejected": mismatch == [],
            "provenance_read_traced": len(provenance_traces) == 1,
            "no_new_semantic_written": semantic_count == 1,
            "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
