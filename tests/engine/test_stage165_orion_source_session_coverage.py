from types import SimpleNamespace

from civilization.engine.stages.stage73_orion_memory_kernel import OrionMemoryStore
from civilization.engine.stages.stage165_orion_source_session_coverage import (
    reorder_for_source_session_coverage,
)


def _write_trip(
    store: OrionMemoryStore,
    source_session_id: str,
    destination: str,
    time_index: float = 1.0,
):
    return store.write_cell(
        memory_system="episodic",
        summary="camping trips",
        content=f"Camping trip to {destination}.",
        source="conversation",
        time_index=time_index,
        metadata={
            "observation_date": "2024-03-01",
            "source_session_id": source_session_id,
        },
    )


def test_stage165_temporal_read_keeps_one_cell_per_source_session_first() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    first = _write_trip(store, "session-1", "Yosemite")
    repeat = _write_trip(store, "session-1", "Zion")
    other_session = _write_trip(store, "session-2", "Bryce")

    results = store.read("How many camping trips did I take?", limit=3)

    assert [item.cell.cell_id for item in results] == [
        first.cell_id,
        other_session.cell_id,
        repeat.cell_id,
    ]
    assert store.trace_events[-1].details["source_session_coverage_applied"] is True


def test_stage165_latest_queries_preserve_existing_order() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    older = _write_trip(store, "session-1", "Yosemite", time_index=1.0)
    newer = _write_trip(store, "session-2", "Zion", time_index=2.0)

    results = store.read("What was my latest camping trip?", limit=2)

    assert [item.cell.cell_id for item in results] == [newer.cell_id, older.cell_id]
    assert store.trace_events[-1].details["source_session_coverage_applied"] is False


def test_stage165_cells_without_session_metadata_keep_their_order() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    cells = [
        store.write_cell(
            memory_system="episodic",
            summary="camping trips",
            content=f"Camping trip {index}.",
            source="conversation",
            metadata={"observation_date": "2024-03-01"},
        )
        for index in range(3)
    ]

    original = store.read("How many camping trips did I take?", limit=3)
    reordered, applied = reorder_for_source_session_coverage(
        original, "How many camping trips did I take?"
    )

    assert [item.cell.cell_id for item in reordered] == [item.cell.cell_id for item in original]
    assert {item.cell.cell_id for item in original} == {cell.cell_id for cell in cells}
    assert applied is False


def test_stage165_limits_source_session_displacement_to_two_ranks() -> None:
    rows = [
        SimpleNamespace(cell=SimpleNamespace(metadata={"source_session_id": session_id}))
        for session_id in ("session-1", "session-1", "session-1", "session-2", "session-3")
    ]

    reordered, applied = reorder_for_source_session_coverage(
        rows, "How many camping trips did I take?"
    )

    assert [row.cell.metadata["source_session_id"] for row in reordered] == [
        "session-1",
        "session-2",
        "session-3",
        "session-1",
        "session-1",
    ]
    assert applied is True
