from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .stage73_orion_memory_kernel import MemoryReadResult, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage79_orion_global_retrieval")


@dataclass(frozen=True)
class OrionTieredReadResult:
    tier: str
    result: MemoryReadResult

    def to_dict(self) -> dict:
        return {"tier": self.tier, **self.result.to_dict()}


class OrionGlobalRetrievalRouter:
    def read(self, session_store: OrionMemoryStore, global_store: OrionMemoryStore, *, query: str, include_global: bool = False, limit: int = 4) -> list[OrionTieredReadResult]:
        session = [OrionTieredReadResult("session", item) for item in session_store.read(query, limit=limit)]
        global_results = [OrionTieredReadResult("global", item) for item in global_store.read(query, limit=limit)] if include_global else []
        ordered = sorted([*session, *global_results], key=lambda item: (item.tier != "session", -item.result.score, item.result.cell.cell_id))
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
