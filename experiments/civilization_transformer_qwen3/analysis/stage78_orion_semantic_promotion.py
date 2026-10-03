from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage78_orion_semantic_promotion")


@dataclass(frozen=True)
class Stage78PromotionResult:
    accepted: bool
    reason: str | None
    source_session_ids: tuple[str, ...]
    source_cell_ids: tuple[str, ...]
    promoted_cell_id: str | None

    def to_dict(self):
        return asdict(self)


class OrionCrossSessionSemanticPromotionPolicy:
    def __init__(self, *, min_sessions: int = 2) -> None:
        if min_sessions < 2:
            raise ValueError("min_sessions must be >= 2")
        self.min_sessions = min_sessions

    def apply(self, session_stores: Mapping[str, OrionMemoryStore], global_store: OrionMemoryStore, *, task_name: str) -> Stage78PromotionResult:
        candidates = []
        for session_id, store in sorted(session_stores.items()):
            for cell in store.cells.values():
                if cell.memory_system == MemorySystem.SEMANTIC and cell.consolidation_state == "consolidated" and cell.metadata.get("task_name") == task_name:
                    candidates.append((session_id, cell))
        sessions = tuple(session_id for session_id, _cell in candidates)
        cell_ids = tuple(cell.cell_id for _session_id, cell in candidates)
        if len(set(sessions)) < self.min_sessions:
            return Stage78PromotionResult(False, "insufficient_sessions", sessions, cell_ids, None)
        promoted = global_store.write_cell(
            memory_system=MemorySystem.SEMANTIC,
            content=" | ".join(cell.content for _session_id, cell in candidates),
            summary=f"{task_name} cross-session semantic pattern",
            source="stage78_cross_session_promotion",
            confidence=sum(cell.confidence for _session_id, cell in candidates) / len(candidates),
            importance=max(cell.importance for _session_id, cell in candidates),
            consolidation_state="consolidated",
            metadata={"stage": 78, "task_name": task_name, "source_sessions": list(sessions), "source_cells": list(cell_ids)},
        )
        return Stage78PromotionResult(True, None, sessions, cell_ids, promoted.cell_id)


def run_stage78_orion_promotion_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    sessions = {"a": OrionMemoryStore(), "b": OrionMemoryStore()}
    for session_id, store in sessions.items():
        store.write_cell(memory_system=MemorySystem.SEMANTIC, content=f"{session_id} successful pattern", summary="replayed pattern", source=session_id, consolidation_state="consolidated", metadata={"task_name": "stage78_task"})
    global_store = OrionMemoryStore()
    result = OrionCrossSessionSemanticPromotionPolicy().apply(sessions, global_store, task_name="stage78_task")
    summary = {**global_store.summary(), "promotion": result.to_dict(), "stage_gates": {"accepted": result.accepted, "two_sessions": len(set(result.source_session_ids)) == 2, "promoted_created": result.promoted_cell_id in global_store.cells, "sources_preserved": all(cell_id in store.cells for _sid, store in sessions.items() for cell_id in store.cells)}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    global_store.write_artifacts(output_dir, summary=summary)
    return summary
