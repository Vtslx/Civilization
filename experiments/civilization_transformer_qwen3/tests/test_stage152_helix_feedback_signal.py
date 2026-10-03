from __future__ import annotations

import json

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage152_helix_feedback_signal import (
    FEEDBACK_ACTIONS,
    APPROVED_MAGNITUDE,
    FAILURE_MAGNITUDE,
    HIT_MAGNITUDE,
    MISS_MAGNITUDE,
    REJECTED_MAGNITUDE,
    SUCCESS_MAGNITUDE,
    FeedbackKind,
    FeedbackSignal,
    HelixFeedbackSignalCollector,
    run_stage152_helix_feedback_signal_smoke,
)
from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import (
    MemoryLinkType,
    MemoryTraceAction,
    MemoryTraceEvent,
    OrionMemoryStore,
)


def test_stage152_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage152_helix_feedback_signal_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage152_helix_feedback_signal"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "feedback_signals.json").exists()


def test_stage152_all_six_kinds_observed(tmp_path) -> None:
    run_stage152_helix_feedback_signal_smoke(output_dir=tmp_path)
    signals = json.loads((tmp_path / "feedback_signals.json").read_text(encoding="utf-8"))
    kinds = {s["kind"] for s in signals}
    assert kinds == {kind.value for kind in FeedbackKind}
    # Each kind appears at least once.
    for kind in FeedbackKind:
        assert any(s["kind"] == kind.value for s in signals)


def test_stage152_collector_is_read_only() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    a = store.write_cell(memory_system="episodic", content="alpha beta", summary="a", source="t")
    b = store.write_cell(memory_system="episodic", content="beta gamma", summary="b", source="t")
    store.link_cells(a.cell_id, b.cell_id, link_type="temporal", weight=1.0)
    store.read("alpha", memory_system="episodic")

    cells_before = [c.to_dict() for c in store.cells.values()]
    links_before = [lk.to_dict() for lk in store.links]
    trace_before = [e.to_dict() for e in store.trace_events]

    HelixFeedbackSignalCollector().collect(store)

    assert [c.to_dict() for c in store.cells.values()] == cells_before
    assert [lk.to_dict() for lk in store.links] == links_before
    assert [e.to_dict() for e in store.trace_events] == trace_before


def test_stage152_collection_is_replayable() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    a = store.write_cell(memory_system="episodic", content="alpha", summary="a", source="t")
    b = store.write_cell(memory_system="episodic", content="beta", summary="b", source="t")
    store.link_cells(a.cell_id, b.cell_id, link_type="temporal", weight=1.0)
    store.read("alpha", memory_system="episodic")

    collector = HelixFeedbackSignalCollector()
    first = collector.collect(store)
    second = collector.collect(store)

    assert [s.to_dict() for s in second] == [s.to_dict() for s in first]
    # signal ids are stable across replays.
    assert [s.signal_id for s in second] == [s.signal_id for s in first]


def test_stage152_read_hit_emits_signal_per_incident_link() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    a = store.write_cell(memory_system="episodic", content="alpha", summary="a", source="t")
    b = store.write_cell(memory_system="episodic", content="beta", summary="b", source="t")
    store.link_cells(a.cell_id, b.cell_id, link_type="temporal", weight=1.0)
    # No system filter: Orion returns only lexically matched cells, so only ``a``
    # is retrieved and its single incident temporal link yields one HIT signal.
    store.read("alpha")

    signals = HelixFeedbackSignalCollector().collect(store)
    hits = [s for s in signals if s.kind is FeedbackKind.HIT]
    assert len(hits) == 1
    assert hits[0].magnitude == pytest.approx(HIT_MAGNITUDE)
    assert hits[0].source_path_id  # non-empty: points at a real link
    assert hits[0].evidence_trace_id.startswith("trace-")


def test_stage152_read_miss_is_query_level_with_empty_path() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    store.write_cell(memory_system="episodic", content="alpha", summary="a", source="t")
    store.read("zzz_no_match_zzz")

    signals = HelixFeedbackSignalCollector().collect(store)
    misses = [s for s in signals if s.kind is FeedbackKind.MISS]
    assert len(misses) == 1
    assert misses[0].source_path_id == ""
    assert misses[0].magnitude == pytest.approx(MISS_MAGNITUDE)


def test_stage152_consolidation_approved_and_procedural_outcomes() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    ep = store.write_cell(memory_system="episodic", content="episode", summary="e", source="t")
    semantic = store.consolidate_episodic_to_semantic([ep.cell_id], summary="s", content="semantic")
    ok = store.write_procedural_from_task_trace(task_name="ok", steps=["x"], outcome="success", source="t")
    bad = store.write_procedural_from_task_trace(task_name="bad", steps=["y"], outcome="failure", source="t")
    store.link_cells(semantic.cell_id, ok.cell_id, link_type="procedure", weight=1.0)
    store.link_cells(semantic.cell_id, bad.cell_id, link_type="procedure", weight=1.0)

    signals = HelixFeedbackSignalCollector().collect(store)
    kinds = {s.kind for s in signals}
    assert FeedbackKind.APPROVED in kinds
    assert FeedbackKind.SUCCESS in kinds
    assert FeedbackKind.FAILURE in kinds
    approved = [s for s in signals if s.kind is FeedbackKind.APPROVED]
    assert approved and approved[0].magnitude == pytest.approx(APPROVED_MAGNITUDE)
    success = [s for s in signals if s.kind is FeedbackKind.SUCCESS]
    assert success and success[0].magnitude == pytest.approx(SUCCESS_MAGNITUDE)
    failure = [s for s in signals if s.kind is FeedbackKind.FAILURE]
    assert failure and failure[0].magnitude == pytest.approx(FAILURE_MAGNITUDE)


def test_stage152_conflict_emits_rejected_signal() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    left = store.write_cell(memory_system="semantic", content="claim left", summary="l", source="t")
    right = store.write_cell(memory_system="semantic", content="claim right", summary="r", source="t")
    store.mark_conflict(left.cell_id, right.cell_id, reason="dispute")

    signals = HelixFeedbackSignalCollector().collect(store)
    rejected = [s for s in signals if s.kind is FeedbackKind.REJECTED]
    assert len(rejected) == 1
    assert rejected[0].magnitude == pytest.approx(REJECTED_MAGNITUDE)


def test_stage152_magnitudes_are_bounded() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    a = store.write_cell(memory_system="episodic", content="alpha", summary="a", source="t")
    b = store.write_cell(memory_system="episodic", content="beta", summary="b", source="t")
    store.link_cells(a.cell_id, b.cell_id, link_type="temporal", weight=1.0)
    store.read("alpha", memory_system="episodic")
    store.read("nope")

    signals = HelixFeedbackSignalCollector().collect(store)
    assert signals
    assert all(-1.0 <= s.magnitude <= 1.0 for s in signals)


def test_stage152_fail_closed_on_dangling_cell_reference() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    # Inject a READ trace that references a cell that does not exist.
    store.trace_events.append(
        MemoryTraceEvent(
            event_id="trace-bogus",
            action=MemoryTraceAction.READ,
            cell_ids=("episodic-999999",),
            created_at=0.0,
            details={"query": "ghost", "result_count": 1, "read_by_id": False},
        )
    )
    with pytest.raises(KeyError):
        HelixFeedbackSignalCollector().collect(store)


def test_stage152_fail_closed_on_conflict_without_link() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    left = store.write_cell(memory_system="semantic", content="l", summary="l", source="t")
    right = store.write_cell(memory_system="semantic", content="r", summary="r", source="t")
    # Inject a CONFLICT trace with no matching conflict link in the store.
    store.trace_events.append(
        MemoryTraceEvent(
            event_id="trace-conflict-bogus",
            action=MemoryTraceAction.CONFLICT,
            cell_ids=(left.cell_id, right.cell_id),
            created_at=0.0,
            details={"link_type": "conflict", "weight": 1.0},
        )
    )
    with pytest.raises(KeyError):
        HelixFeedbackSignalCollector().collect(store)


def test_stage152_non_feedback_actions_are_skipped() -> None:
    store = OrionMemoryStore(now_fn=lambda: 0.0)
    store.write_cell(memory_system="working", content="scratch", summary="s", source="t")
    # Only WRITE/LINK-less traces exist; no feedback-relevant events.
    signals = HelixFeedbackSignalCollector().collect(store)
    assert signals == []
    # FEEDBACK_ACTIONS is exactly the feedback-relevant set.
    assert FEEDBACK_ACTIONS == frozenset(
        {MemoryTraceAction.READ, MemoryTraceAction.CONSOLIDATE, MemoryTraceAction.CONFLICT}
    )


def test_stage152_signal_round_trip_serialization() -> None:
    signal = FeedbackSignal(
        signal_id="signal-000001",
        source_path_id="link:episodic-000001:episodic-000002:temporal",
        kind=FeedbackKind.HIT,
        magnitude=0.5,
        evidence_trace_id="trace-000003",
        created_at=4003.0,
        details={"retrieved_cell_id": "episodic-000001", "link_type": "temporal"},
    )
    restored = FeedbackSignal.from_dict(json.loads(json.dumps(signal.to_dict(), ensure_ascii=False)))
    assert restored == signal
    assert isinstance(restored.kind, FeedbackKind)


def test_stage152_link_type_enum_values_match_trace_usage() -> None:
    # Guards against a silent drift between the collector's expected link types
    # and the kernel's actual enum values.
    assert MemoryLinkType.REPLAY.value == "replay"
    assert MemoryLinkType.PROCEDURE.value == "procedure"
    assert MemoryLinkType.CONFLICT.value == "conflict"
