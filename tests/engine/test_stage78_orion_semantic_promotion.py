from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from civilization.engine.stages.stage78_orion_semantic_promotion import OrionCrossSessionSemanticPromotionPolicy, run_stage78_orion_promotion_smoke


def _store(session: str):
    store = OrionMemoryStore()
    store.write_cell(memory_system=MemorySystem.SEMANTIC, content=session, summary=session, source=session, consolidation_state="consolidated", metadata={"task_name": "task"})
    return store


def test_stage78_promotes_only_after_two_sessions() -> None:
    global_store = OrionMemoryStore()
    rejected = OrionCrossSessionSemanticPromotionPolicy().apply({"a": _store("a")}, global_store, task_name="task")
    accepted = OrionCrossSessionSemanticPromotionPolicy().apply({"a": _store("a"), "b": _store("b")}, global_store, task_name="task")
    assert not rejected.accepted and rejected.reason == "insufficient_sessions"
    assert accepted.accepted and accepted.promoted_cell_id in global_store.cells


def test_stage78_smoke_writes_artifacts(tmp_path) -> None:
    summary = run_stage78_orion_promotion_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert (tmp_path / "memory_cells.json").exists()
