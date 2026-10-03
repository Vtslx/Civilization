from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage134_eagle_skill_candidates import EagleSkillCandidateBuilder
from .stage135_eagle_approved_skill import EagleApprovedSkillConsolidator, _sample_traces, _write_trace_cells
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage136_eagle_skill_retrieval")


@dataclass(frozen=True)
class EagleSkillRecall:
    procedural_cell_id: str
    task_signature: tuple[str, str, str]
    strategy_steps: tuple[str, ...]
    support_trace_ids: tuple[str, ...]
    success_count: int
    failure_count: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_signature"] = list(self.task_signature)
        payload["strategy_steps"] = list(self.strategy_steps)
        payload["support_trace_ids"] = list(self.support_trace_ids)
        return payload


class EagleSkillRetriever:
    """Retrieves approved procedural skills with trace-backed provenance."""

    def retrieve(self, store: OrionMemoryStore, query: str) -> list[EagleSkillRecall]:
        if not query.strip():
            raise ValueError("query must not be empty")
        recalls: list[EagleSkillRecall] = []
        for result in store.read(query, memory_system=MemorySystem.PROCEDURAL):
            skill = result.cell
            if not result.matched_terms or skill.source != "eagle_stage135":
                continue
            signature = skill.metadata.get("eagle_task_signature")
            steps = skill.metadata.get("eagle_strategy_steps")
            support_ids = skill.metadata.get("eagle_support_trace_ids")
            if not isinstance(signature, list) or len(signature) != 3 or not isinstance(steps, list) or not isinstance(support_ids, list):
                continue
            links = [link for link in store.links if link.link_type == MemoryLinkType.PROCEDURE and link.target_cell_id == skill.cell_id]
            linked_trace_ids = {link.metadata.get("eagle_trace_id") for link in links}
            if linked_trace_ids != set(support_ids):
                continue
            source_ids = [link.source_cell_id for link in links]
            evidence = store.read_cells(source_ids, trace_details={"eagle_skill_retrieval": True, "skill_cell_id": skill.cell_id, "query": query})
            if len(evidence) != len(source_ids) or any(cell.metadata.get("eagle_trace_id") not in support_ids for cell in evidence):
                continue
            recalls.append(EagleSkillRecall(skill.cell_id, tuple(signature), tuple(steps), tuple(support_ids), len(skill.metadata.get("eagle_success_trace_ids", [])), len(skill.metadata.get("eagle_failure_trace_ids", [])), skill.confidence))
        return recalls


def run_stage136_eagle_skill_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store = OrionMemoryStore()
    traces = _sample_traces()
    candidate = EagleSkillCandidateBuilder().build(traces)[0]
    trace_cells = _write_trace_cells(store, traces)
    skill = EagleApprovedSkillConsolidator().consolidate(store, candidate, trace_cells, approval_id="stage136").procedural_cell_id
    retriever = EagleSkillRetriever()
    recalls = retriever.retrieve(store, "collect_sample laboratory")
    mismatch = retriever.retrieve(store, "repair_robot workshop")
    summary = {
        "stage": "stage136_eagle_skill_retrieval",
        "recalls": [recall.to_dict() for recall in recalls],
        "stage_gates": {
            "skill_recalled": len(recalls) == 1 and recalls[0].procedural_cell_id == skill,
            "strategy_recalled": len(recalls) == 1 and recalls[0].strategy_steps == candidate.strategy_steps,
            "outcome_counts_preserved": len(recalls) == 1 and recalls[0].success_count == 3 and recalls[0].failure_count == 1,
            "trace_links_verified": len(recalls) == 1 and recalls[0].support_trace_ids == candidate.support_trace_ids,
            "mismatch_rejected": mismatch == [],
            "retrieval_traced": any(event.details.get("eagle_skill_retrieval") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
