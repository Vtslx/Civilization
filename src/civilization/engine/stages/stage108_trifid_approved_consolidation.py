from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage106_trifid_replay_scheduler import TrifidReplayScheduler
from .stage107_trifid_consolidation_candidates import (
    TrifidConsolidationCandidate,
    TrifidConsolidationCandidateBuilder,
)
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage108_trifid_approved_consolidation")


@dataclass(frozen=True)
class TrifidConsolidationResult:
    candidate_fingerprint: str
    semantic_cell_id: str
    created: bool
    approval_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrifidApprovedConsolidator:
    """Consolidates a replay-evidence candidate only after explicit approval."""

    def consolidate(
        self,
        store: OrionMemoryStore,
        candidate: TrifidConsolidationCandidate,
        *,
        approval_id: str,
    ) -> TrifidConsolidationResult:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty")
        fingerprint = self._fingerprint(candidate)
        existing = next(
            (
                cell
                for cell in store.cells.values()
                if cell.memory_system == MemorySystem.SEMANTIC
                and cell.metadata.get("trifid_candidate_fingerprint") == fingerprint
            ),
            None,
        )
        if existing is not None:
            return TrifidConsolidationResult(fingerprint, existing.cell_id, False, approval_id)
        semantic = store.consolidate_episodic_to_semantic(
            list(candidate.anchor_cell_ids),
            summary=f"Trifid {candidate.signature[0]} {candidate.signature[2]} pattern",
            content=" | ".join(candidate.episode_ids),
            source="trifid_stage108",
            confidence=candidate.average_confidence,
            importance=candidate.average_importance,
            metadata={
                "trifid_candidate_fingerprint": fingerprint,
                "trifid_candidate_episode_ids": list(candidate.episode_ids),
                "trifid_candidate_signature": list(candidate.signature),
                "trifid_approval_id": approval_id,
            },
        )
        return TrifidConsolidationResult(fingerprint, semantic.cell_id, True, approval_id)

    @staticmethod
    def _fingerprint(candidate: TrifidConsolidationCandidate) -> str:
        return "trifid:" + "|".join(candidate.episode_ids)


def run_stage108_trifid_approved_consolidation_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    source_ids: list[str] = []
    for start in (100.0, 200.0, 300.0):
        arrival = store.write_cell(memory_system="episodic", content="robot entered laboratory", summary="arrival", source="observation", time_index=start)
        collection = store.write_cell(memory_system="working", content="robot collected sample", summary="collection", source="observation", time_index=start + 2.0, ttl_seconds=60.0)
        binder.bind(store, source_cell_ids=[arrival.cell_id, collection.cell_id], scene="laboratory", entities=["robot"], goal="collect sample", action="inspect", outcome="secured", max_source_gap_seconds=5.0)
        source_ids.extend([arrival.cell_id, collection.cell_id])
    scheduled = TrifidReplayScheduler().schedule_and_replay(store, binder, "robot laboratory", time_index=201.0, window_seconds=150.0, budget=2)
    candidate = TrifidConsolidationCandidateBuilder().build(store, binder, scheduled)[0]
    consolidator = TrifidApprovedConsolidator()
    approval_rejected = False
    try:
        consolidator.consolidate(store, candidate, approval_id="")
    except ValueError:
        approval_rejected = True
    first = consolidator.consolidate(store, candidate, approval_id="stage108-smoke-approval")
    second = consolidator.consolidate(store, candidate, approval_id="stage108-repeat-approval")
    semantic = store.cells[first.semantic_cell_id]
    consolidation_events = [event for event in store.trace_events if event.action.value == "consolidate"]
    summary = {
        "stage": "stage108_trifid_approved_consolidation",
        "result": first.to_dict(),
        "stage_gates": {
            "approval_required": approval_rejected,
            "semantic_created": first.created and semantic.memory_system == MemorySystem.SEMANTIC,
            "candidate_provenance_preserved": semantic.metadata.get("trifid_candidate_episode_ids") == list(candidate.episode_ids),
            "repeat_is_idempotent": not second.created and second.semantic_cell_id == first.semantic_cell_id,
            "orion_consolidation_traced": len(consolidation_events) == 1,
            "anchor_cells_preserved": all(anchor_id in store.cells for anchor_id in candidate.anchor_cell_ids),
            "source_cells_preserved": all(cell_id in store.cells for cell_id in source_ids),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
