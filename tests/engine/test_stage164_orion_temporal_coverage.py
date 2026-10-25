from __future__ import annotations

from civilization.engine.stages.stage73_orion_memory_kernel import OrionMemoryStore
from civilization.engine.stages.stage164_orion_temporal_coverage import reorder_for_temporal_coverage


def _write_trip(store: OrionMemoryStore, day: int, destination: str):
    return store.write_cell(
        memory_system="episodic",
        summary="camping trips",
        content=f"Camping trips to {destination}.",
        source="conversation",
        time_index=float(day),
        metadata={"observation_date": f"2024-03-{day:02d}"},
    )


def test_stage164_temporal_count_query_spreads_top_results_across_dates() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    first = _write_trip(store, 1, "Yosemite")
    duplicate = _write_trip(store, 1, "California")
    second = _write_trip(store, 2, "Zion")
    third = _write_trip(store, 3, "Bryce")

    results = store.read("How many camping trips did I take?", limit=3)

    assert [item.cell.cell_id for item in results] == [first.cell_id, second.cell_id, third.cell_id]
    assert store.trace_events[-1].details["temporal_coverage_applied"] is True
    assert duplicate.cell_id not in {item.cell.cell_id for item in results}


def test_stage164_latest_query_keeps_recency_order_without_date_diversity() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    older = _write_trip(store, 1, "Yosemite")
    newer = _write_trip(store, 2, "Zion")

    results = store.read("What is my latest camping trip?", limit=2)

    assert [item.cell.cell_id for item in results] == [newer.cell_id, older.cell_id]
    assert store.trace_events[-1].details["temporal_coverage_applied"] is False


def test_stage164_non_temporal_query_preserves_relevance_order() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    newest_irrelevant = store.write_cell(
        memory_system="episodic",
        summary="weekend preference note",
        content="A quiet weekend preference note.",
        source="conversation",
        time_index=3.0,
        metadata={"observation_date": "2024-03-03"},
    )
    relevant = store.write_cell(
        memory_system="episodic",
        summary="camping equipment preference",
        content="I prefer a lightweight camping tent.",
        source="conversation",
        time_index=1.0,
        metadata={"observation_date": "2024-03-01"},
    )

    results = store.read("What kind of camping tent do I prefer for my weekend preference?", limit=2)

    assert results[0].cell.cell_id == relevant.cell_id
    assert results[1].cell.cell_id == newest_irrelevant.cell_id
    assert store.trace_events[-1].details["temporal_coverage_applied"] is False


def test_stage164_short_result_sets_are_unchanged() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    cell = _write_trip(store, 1, "Yosemite")

    result = store.read("How many camping trips did I take?", limit=5)

    assert [item.cell.cell_id for item in result] == [cell.cell_id]
    assert store.trace_events[-1].details["temporal_coverage_applied"] is False
