from __future__ import annotations

import json
from pathlib import Path

from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import (
    MemoryLinkType,
    MemorySystem,
    MemoryTraceAction,
    OrionMemoryStore,
    run_stage73_orion_memory_kernel_smoke,
)


def test_stage73_writes_and_reads_all_memory_systems() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    cells = [
        store.write_cell(memory_system=MemorySystem.WORKING, content="focus on active task", summary="active focus", source="test"),
        store.write_cell(memory_system=MemorySystem.EPISODIC, content="event happened in sequence", summary="event sequence", source="test"),
        store.write_cell(memory_system=MemorySystem.SEMANTIC, content="stable concept relation", summary="concept relation", source="test"),
        store.write_cell(memory_system=MemorySystem.PROCEDURAL, content="step one then step two", summary="task procedure", source="test"),
    ]

    for cell in cells:
        results = store.read(cell.summary, memory_system=cell.memory_system)
        assert results
        assert results[0].cell.cell_id == cell.cell_id

    summary = store.summary()
    assert summary["cell_counts"] == {"working": 1, "episodic": 1, "semantic": 1, "procedural": 1}
    assert "write" in summary["trace_actions"]
    assert "read" in summary["trace_actions"]


def test_stage73_working_memory_ttl_expires_and_is_excluded() -> None:
    clock = {"now": 10.0}
    store = OrionMemoryStore(now_fn=lambda: clock["now"])
    cell = store.write_cell(
        memory_system=MemorySystem.WORKING,
        content="temporary working context",
        summary="temporary context",
        source="test",
        ttl_seconds=2.0,
    )

    assert store.read("temporary", memory_system=MemorySystem.WORKING)
    clock["now"] = 13.0
    expired = store.expire_working_memory()

    assert [item.cell_id for item in expired] == [cell.cell_id]
    assert store.cells[cell.cell_id].decay_state == "expired"
    assert store.cells[cell.cell_id].metadata["transfer_candidate"] is True
    assert store.read("temporary", memory_system=MemorySystem.WORKING) == []
    assert any(event.action == MemoryTraceAction.EXPIRE for event in store.trace_events)


def test_stage73_episodic_to_semantic_consolidation_traces_replay() -> None:
    store = OrionMemoryStore(now_fn=lambda: 20.0)
    first = store.write_cell(memory_system="episodic", content="first event", summary="first", source="test", time_index=1.0)
    second = store.write_cell(memory_system="episodic", content="second event", summary="second", source="test", time_index=2.0)

    semantic = store.consolidate_episodic_to_semantic(
        [first.cell_id, second.cell_id],
        summary="stable lesson",
        content="events imply a stable lesson",
    )

    assert semantic.memory_system == MemorySystem.SEMANTIC
    assert semantic.consolidation_state == "consolidated"
    assert semantic.metadata["consolidated_from"] == [first.cell_id, second.cell_id]
    assert store.cells[first.cell_id].consolidation_state == "replayed"
    assert store.cells[second.cell_id].consolidation_state == "replayed"
    assert sum(link.link_type == MemoryLinkType.REPLAY for link in store.links) == 2
    assert any(event.action == MemoryTraceAction.CONSOLIDATE for event in store.trace_events)


def test_stage73_procedural_memory_from_task_trace_preserves_outcome() -> None:
    store = OrionMemoryStore(now_fn=lambda: 30.0)
    cell = store.write_procedural_from_task_trace(
        task_name="repair",
        steps=["inspect", "patch", "test"],
        outcome="success",
        source="test",
    )

    assert cell.memory_system == MemorySystem.PROCEDURAL
    assert cell.metadata["task_name"] == "repair"
    assert cell.metadata["outcome"] == "success"
    assert cell.metadata["steps"] == ["inspect", "patch", "test"]
    assert store.read("repair success", memory_system=MemorySystem.PROCEDURAL)[0].cell.cell_id == cell.cell_id


def test_stage73_conflict_link_preserves_original_cells() -> None:
    store = OrionMemoryStore(now_fn=lambda: 40.0)
    left = store.write_cell(memory_system="semantic", content="current stage avoids service integration", summary="current boundary", source="test")
    right = store.write_cell(memory_system="semantic", content="future stage may integrate service", summary="future integration", source="test")

    link = store.mark_conflict(left.cell_id, right.cell_id, reason="current versus future")

    assert link.link_type == MemoryLinkType.CONFLICT
    assert left.cell_id in store.cells
    assert right.cell_id in store.cells
    assert store.cells[left.cell_id].content == "current stage avoids service integration"
    assert store.cells[right.cell_id].content == "future stage may integrate service"
    assert any(event.action == MemoryTraceAction.CONFLICT for event in store.trace_events)


def test_stage73_smoke_writes_artifacts_and_passes_gate(tmp_path) -> None:
    summary = run_stage73_orion_memory_kernel_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["passes_stage_gate"] is True
    assert len(json.loads((tmp_path / "memory_cells.json").read_text(encoding="utf-8"))) >= 7
    assert Path(tmp_path / "trace_events.jsonl").read_text(encoding="utf-8").strip()
