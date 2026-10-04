from __future__ import annotations

from pathlib import Path

from .stage133_eagle_task_trace import EagleTaskTraceNormalizer
from .stage134_eagle_skill_candidates import EagleSkillCandidateBuilder
from .stage135_eagle_approved_skill import EagleApprovedSkillConsolidator, _sample_traces, _write_trace_cells
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage137_eagle_skill_conflict_guard")


class EagleSkillConflictGuard:
    """Adds conflict links for failed, divergent strategies without overwriting approved skills."""

    def detect_and_mark(self, store: OrionMemoryStore, trace_cell_id: str) -> list[str]:
        trace_cell = store.cells.get(trace_cell_id)
        if trace_cell is None or trace_cell.source != "eagle_task_trace" or trace_cell.memory_system != MemorySystem.EPISODIC:
            raise ValueError("trace_cell_id must reference an Eagle task trace")
        if trace_cell.metadata.get("eagle_outcome") != "failure":
            return []
        signature = trace_cell.metadata.get("eagle_task_signature")
        steps = tuple(trace_cell.metadata.get("eagle_steps", []))
        conflicted: list[str] = []
        for skill in store.cells.values():
            if skill.memory_system != MemorySystem.PROCEDURAL or skill.source != "eagle_stage135" or skill.metadata.get("eagle_task_signature") != signature:
                continue
            if steps == tuple(skill.metadata.get("eagle_strategy_steps", [])):
                continue
            existing = any(link.source_cell_id == trace_cell_id and link.target_cell_id == skill.cell_id and link.link_type == MemoryLinkType.CONFLICT for link in store.links)
            if not existing:
                store.mark_conflict(trace_cell_id, skill.cell_id, reason="eagle_strategy_conflict")
            conflicted.append(skill.cell_id)
        return conflicted


def _new_trace_cell(store: OrionMemoryStore, *, steps: list[str], outcome: str, time_index: float) -> str:
    normalizer = EagleTaskTraceNormalizer()
    trace = normalizer.normalize({"task_name": "collect_sample", "scene": "laboratory", "goal": "secure sample", "steps": steps, "outcome": outcome, "source_cell_ids": [f"external-{time_index}"], "time_index": time_index})
    cell = store.write_cell(memory_system=MemorySystem.EPISODIC, content=" | ".join(trace.steps), summary=f"{trace.task_name} {trace.outcome}", source="eagle_task_trace", time_index=trace.time_index, metadata={"eagle_trace_id": trace.trace_id, "eagle_task_signature": [trace.task_name, trace.scene, trace.goal], "eagle_outcome": trace.outcome, "eagle_steps": list(trace.steps)})
    return cell.cell_id


def run_stage137_eagle_skill_conflict_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore()
    traces = _sample_traces()
    candidate = EagleSkillCandidateBuilder().build(traces)[0]
    trace_cells = _write_trace_cells(store, traces)
    skill = EagleApprovedSkillConsolidator().consolidate(store, candidate, trace_cells, approval_id="stage137").procedural_cell_id
    same_strategy_failure = _new_trace_cell(store, steps=list(candidate.strategy_steps), outcome="failure", time_index=300.0)
    divergent_failure = _new_trace_cell(store, steps=["inspect bench", "drop sample"], outcome="failure", time_index=330.0)
    skill_before = store.cells[skill].to_dict()
    same_result = EagleSkillConflictGuard().detect_and_mark(store, same_strategy_failure)
    first_result = EagleSkillConflictGuard().detect_and_mark(store, divergent_failure)
    link_count = len(store.links)
    second_result = EagleSkillConflictGuard().detect_and_mark(store, divergent_failure)
    conflict_links = [link for link in store.links if link.link_type == MemoryLinkType.CONFLICT]
    summary = {
        "stage": "stage137_eagle_skill_conflict_guard",
        "stage_gates": {
            "same_strategy_failure_not_misclassified": same_result == [],
            "divergent_failure_conflicted": first_result == [skill] and len(conflict_links) == 1,
            "conflict_is_idempotent": second_result == [skill] and len(store.links) == link_count,
            "approved_skill_unchanged": store.cells[skill].to_dict() == skill_before,
            "conflict_reason_traced": conflict_links[0].metadata.get("reason") == "eagle_strategy_conflict" and any(event.action.value == "conflict" for event in store.trace_events),
            "failed_trace_cells_preserved": same_strategy_failure in store.cells and divergent_failure in store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
