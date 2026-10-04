from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore
from civilization.engine.stages.stage76_orion_replay_consolidation import OrionReplayConsolidationPolicy, run_stage76_orion_replay_smoke


def _episode(store, outcome: str) -> None:
    store.write_cell(memory_system=MemorySystem.EPISODIC, content=outcome, summary=outcome, source="test", metadata={"task_name": "task", "outcome": outcome})


def test_stage76_consolidates_success_and_preserves_failure_procedure() -> None:
    store = OrionMemoryStore()
    _episode(store, "success")
    _episode(store, "success")
    _episode(store, "failure")
    result = OrionReplayConsolidationPolicy().apply(store, source="test", task_name="task")
    assert result.accepted
    assert store.cells[result.semantic_cell_id].memory_system == MemorySystem.SEMANTIC
    assert len(result.procedural_cell_ids) == 2
    assert all(cell_id in store.cells for cell_id in result.source_episode_ids)


def test_stage76_rejects_without_mutating_when_episodes_are_insufficient() -> None:
    store = OrionMemoryStore()
    _episode(store, "success")
    before = store.summary().copy()
    result = OrionReplayConsolidationPolicy().apply(store, source="test", task_name="task")
    assert not result.accepted and result.reason == "insufficient_episodes"
    assert store.summary()["cell_counts"] == before["cell_counts"]


def test_stage76_smoke_writes_artifacts(tmp_path) -> None:
    summary = run_stage76_orion_replay_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"]
    assert (tmp_path / "memory_cells.json").exists()
