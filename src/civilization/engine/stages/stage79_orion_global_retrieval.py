from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryReadResult, OrionMemoryStore
from .stage161_orion_relevance_ranking import OrionRelevanceRanker, has_recency_intent, tokenize
from .stage163_orion_role_aware_ranking import OrionRoleAwareRanker
from .stage164_orion_temporal_coverage import reorder_for_temporal_coverage
from .stage165_orion_source_session_coverage import reorder_for_source_session_coverage


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage79_orion_global_retrieval")


@dataclass(frozen=True)
class OrionTieredReadResult:
    tier: str
    result: MemoryReadResult

    def to_dict(self) -> dict:
        return {"tier": self.tier, **self.result.to_dict()}


class OrionGlobalRetrievalRouter:
    def __init__(self, *, ranker: OrionRelevanceRanker | None = None) -> None:
        self._ranker = ranker if ranker is not None else OrionRoleAwareRanker()

    def read(self, session_store: OrionMemoryStore, global_store: OrionMemoryStore, *, query: str, include_global: bool = False, limit: int = 4) -> list[OrionTieredReadResult]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        session_store.expire_working_memory()
        session_cells = tuple(cell for cell in session_store.cells.values() if cell.decay_state == "active")
        if include_global:
            global_store.expire_working_memory()
            global_cells = tuple(cell for cell in global_store.cells.values() if cell.decay_state == "active")
        else:
            global_cells = ()
        session = [
            OrionTieredReadResult("session", item)
            for item in session_store.read(query, limit=limit, ranking_context=global_cells, ranker=self._ranker)
        ]
        global_results = (
            [
                OrionTieredReadResult("global", item)
                for item in global_store.read(query, limit=limit, ranking_context=session_cells, ranker=self._ranker)
            ]
            if include_global
            else []
        )
        recent_first = has_recency_intent(query, tokenize(query))

        def order_key(item: OrionTieredReadResult) -> tuple[float, float, bool, str]:
            try:
                time_index = float(item.result.cell.time_index)
            except (TypeError, ValueError):
                time_index = 0.0
            if not math.isfinite(time_index):
                time_index = 0.0
            temporal_key = -time_index if recent_first else time_index
            return (-item.result.score, temporal_key, item.tier != "session", item.result.cell.cell_id)

        ordered = sorted([*session, *global_results], key=order_key)
        ordered, _ = reorder_for_temporal_coverage(ordered, query)
        ordered, _ = reorder_for_source_session_coverage(ordered, query)
        return ordered[:limit]


def run_stage79_orion_global_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    session.write_cell(memory_system="semantic", content="session local evidence", summary="session evidence", source="session")
    global_store.write_cell(memory_system="semantic", content="global promoted evidence", summary="global evidence", source="stage78")
    router = OrionGlobalRetrievalRouter()
    local = router.read(session, global_store, query="evidence", include_global=False)
    combined = router.read(session, global_store, query="evidence", include_global=True)
    summary = {**global_store.summary(), "local": [item.to_dict() for item in local], "combined": [item.to_dict() for item in combined], "stage_gates": {"global_opt_in": all(item.tier == "session" for item in local), "global_visible_when_enabled": any(item.tier == "global" for item in combined), "session_priority": combined[0].tier == "session"}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    global_store.write_artifacts(output_dir, summary=summary)
    return summary
