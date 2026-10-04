from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .stage133_eagle_task_trace import EagleTaskTrace, EagleTaskTraceNormalizer
from .stage134_eagle_skill_candidates import EagleSkillCandidate, EagleSkillCandidateBuilder
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage135_eagle_approved_skill")


@dataclass(frozen=True)
class EagleSkillConsolidationResult:
    procedural_cell_id: str
    fingerprint: str
    created: bool

    def to_dict(self) -> dict:
        return asdict(self)


class EagleApprovedSkillConsolidator:
    """Writes a procedural skill only after explicit approval and keeps trace provenance."""

    def consolidate(self, store: OrionMemoryStore, candidate: EagleSkillCandidate, trace_cells: dict[str, str], *, approval_id: str) -> EagleSkillConsolidationResult:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty")
        if any(trace_id not in trace_cells for trace_id in candidate.support_trace_ids):
            raise ValueError("candidate support trace is missing")
        existing = next((cell for cell in store.cells.values() if cell.memory_system == MemorySystem.PROCEDURAL and cell.metadata.get("eagle_candidate_fingerprint") == candidate.fingerprint), None)
        if existing is not None:
            return EagleSkillConsolidationResult(existing.cell_id, candidate.fingerprint, False)
        skill = store.write_procedural_from_task_trace(
            task_name=candidate.task_signature[0],
            steps=list(candidate.strategy_steps),
            outcome="success",
            source="eagle_stage135",
            confidence=candidate.confidence,
            importance=candidate.importance,
            metadata={
                "eagle_candidate_fingerprint": candidate.fingerprint,
                "eagle_task_signature": list(candidate.task_signature),
                "eagle_strategy_steps": list(candidate.strategy_steps),
                "eagle_support_trace_ids": list(candidate.support_trace_ids),
                "eagle_success_trace_ids": list(candidate.success_trace_ids),
                "eagle_failure_trace_ids": list(candidate.failure_trace_ids),
                "eagle_approval_id": approval_id,
            },
        )
        for trace_id in candidate.support_trace_ids:
            store.link_cells(trace_cells[trace_id], skill.cell_id, link_type=MemoryLinkType.PROCEDURE, metadata={"eagle_candidate_fingerprint": candidate.fingerprint, "eagle_trace_id": trace_id})
        return EagleSkillConsolidationResult(skill.cell_id, candidate.fingerprint, True)


def _sample_traces() -> list[EagleTaskTrace]:
    normalizer = EagleTaskTraceNormalizer()
    raw = [
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-001"], "time_index": 100.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-002"], "time_index": 130.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "grasp sample", "seal sample"], "outcome": "success", "source_cell_ids": ["episodic-003"], "time_index": 160.0},
        {"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": ["inspect bench", "drop sample"], "outcome": "failure", "source_cell_ids": ["episodic-004"], "time_index": 200.0},
    ]
    return [normalizer.normalize(item) for item in raw]


def _write_trace_cells(store: OrionMemoryStore, traces: list[EagleTaskTrace]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for trace in traces:
        cell = store.write_cell(memory_system=MemorySystem.EPISODIC, content=" | ".join(trace.steps), summary=f"{trace.task_name} {trace.outcome}", source="eagle_task_trace", time_index=trace.time_index, metadata={"eagle_trace_id": trace.trace_id, "eagle_task_signature": [trace.task_name, trace.scene, trace.goal], "eagle_outcome": trace.outcome, "eagle_steps": list(trace.steps)})
        mapping[trace.trace_id] = cell.cell_id
    return mapping


def run_stage135_eagle_approved_skill_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore()
    traces = _sample_traces()
    candidate = EagleSkillCandidateBuilder().build(traces)[0]
    trace_cells = _write_trace_cells(store, traces)
    consolidator = EagleApprovedSkillConsolidator()
    approval_rejected = False
    try:
        consolidator.consolidate(store, candidate, trace_cells, approval_id="")
    except ValueError:
        approval_rejected = True
    first = consolidator.consolidate(store, candidate, trace_cells, approval_id="stage135-approved")
    second = consolidator.consolidate(store, candidate, trace_cells, approval_id="stage135-repeat")
    skill = store.cells[first.procedural_cell_id]
    procedure_links = [link for link in store.links if link.link_type == MemoryLinkType.PROCEDURE and link.target_cell_id == skill.cell_id]
    summary = {
        "stage": "stage135_eagle_approved_skill",
        "result": first.to_dict(),
        "stage_gates": {
            "approval_required": approval_rejected,
            "procedural_skill_created": first.created and skill.memory_system == MemorySystem.PROCEDURAL,
            "strategy_metadata_preserved": skill.metadata.get("eagle_strategy_steps") == list(candidate.strategy_steps),
            "trace_provenance_linked": len(procedure_links) == len(candidate.support_trace_ids),
            "failure_evidence_preserved": skill.metadata.get("eagle_failure_trace_ids") == list(candidate.failure_trace_ids),
            "repeat_is_idempotent": not second.created and second.procedural_cell_id == first.procedural_cell_id,
            "source_trace_cells_preserved": all(cell_id in store.cells for cell_id in trace_cells.values()),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
