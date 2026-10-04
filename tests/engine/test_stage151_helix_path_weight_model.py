from __future__ import annotations

import json

import pytest

from civilization.engine.stages.stage145_rosette_honeycomb_graph import RosetteHoneycombEdge
from civilization.engine.stages.stage151_helix_path_weight_model import (
    HelixWeightStore,
    PathWeight,
    WeightUpdateAction,
    WeightUpdateEvent,
    path_id_for_edge,
    path_id_for_link,
    run_stage151_helix_path_weight_model_smoke,
)
from civilization.engine.stages.stage73_orion_memory_kernel import (
    MemoryLink,
    MemoryLinkType,
)


def _link(weight: float = 1.0, link_type: MemoryLinkType = MemoryLinkType.TEMPORAL) -> MemoryLink:
    return MemoryLink(
        source_cell_id="episodic-000001",
        target_cell_id="episodic-000002",
        link_type=link_type,
        weight=weight,
        created_at=1000.0,
    )


def test_stage151_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage151_helix_path_weight_model_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage151_helix_path_weight_model"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())

    # Artifacts written (real files, not just a summary).
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "path_weights.json").exists()
    assert (tmp_path / "weight_update_events.jsonl").exists()

    # Every required update action appears in the audit trace.
    written_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(written_summary["update_actions"]) >= {a.value for a in WeightUpdateAction}


def test_stage151_initial_weight_equals_baseline_and_version_zero() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    link = _link(weight=0.8)
    edge = RosetteHoneycombEdge("n1", "n2", "consolidation", 1.0)

    link_weight = store.register_link(link)
    edge_weight = store.register_edge(edge)

    assert link_weight.value == pytest.approx(0.8)
    assert link_weight.baseline == pytest.approx(0.8)
    assert link_weight.version == 0
    assert link_weight.update_reason == "init"
    assert link_weight.feedback_source == "baseline"
    assert edge_weight.value == pytest.approx(1.0)
    assert edge_weight.version == 0

    # Registration is idempotent: the same path returns the existing record.
    assert store.register_link(link) is link_weight


def test_stage151_propose_does_not_change_current_weight() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.5))

    event = store.propose(weight.path_id, 0.9, reason="candidate", feedback_source="read_hit")

    assert event.action is WeightUpdateAction.PROPOSE
    assert event.old_value == pytest.approx(0.5)
    assert event.new_value == pytest.approx(0.9)
    assert event.version == 0  # propose does not advance version
    assert store.weights[weight.path_id].value == pytest.approx(0.5)
    assert store.weights[weight.path_id].version == 0


def test_stage151_approve_increments_version_and_preserves_history() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.5))

    updated = store.approve(weight.path_id, 0.9, reason="skill reuse success", feedback_source="skill_success")

    assert updated.version == 1
    assert updated.value == pytest.approx(0.9)
    assert updated.update_reason == "skill reuse success"
    assert updated.feedback_source == "skill_success"
    # The original value survives in the audit trace, not overwritten.
    approve_events = [e for e in store.update_events if e.action is WeightUpdateAction.APPROVE]
    assert any(e.old_value == pytest.approx(0.5) and e.new_value == pytest.approx(0.9) for e in approve_events)


def test_stage151_reject_leaves_weight_untouched() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.7))

    event = store.reject(weight.path_id, 0.1, reason="conflict suppressed", feedback_source="conflict_decision")

    assert event.action is WeightUpdateAction.REJECT
    assert event.new_value == pytest.approx(0.1)
    assert store.weights[weight.path_id].value == pytest.approx(0.7)
    assert store.weights[weight.path_id].version == 0


def test_stage151_decay_moves_toward_baseline() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.4))
    store.approve(weight.path_id, 1.0, reason="boosted", feedback_source="read_hit")

    decayed = store.decay(weight.path_id, decay_rate=0.5)

    # new = 1.0 + 0.5 * (0.4 - 1.0) = 0.7, moved halfway back to baseline 0.4
    assert decayed.value == pytest.approx(0.7)
    assert decayed.version == 2
    assert decayed.update_reason == "decay"


def test_stage151_clip_clamps_and_versions_only_on_change() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.5))

    # No-op clip: value already in range -> version unchanged, event still recorded.
    unchanged = store.clip(weight.path_id, lo=0.0, hi=1.0)
    assert unchanged.value == pytest.approx(0.5)
    assert unchanged.version == 0

    store.approve(weight.path_id, 1.5, reason="overshoot", feedback_source="read_hit")
    clamped = store.clip(weight.path_id, lo=0.0, hi=1.0)
    assert clamped.value == pytest.approx(1.0)
    assert clamped.version == 2  # approve(1) + clip(2)
    assert clamped.update_reason == "clip"


def test_stage151_round_trip_serialization() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.6))
    store.propose(weight.path_id, 0.9, reason="candidate", feedback_source="read_hit")
    store.approve(weight.path_id, 0.8, reason="approved", feedback_source="skill_success")
    store.reject(weight.path_id, 0.1, reason="rejected", feedback_source="conflict_decision")

    payload = json.loads(json.dumps(store.to_dict(), ensure_ascii=False))
    restored = HelixWeightStore.from_dict(payload, now_fn=lambda: 1.0)

    assert [w.to_dict() for w in sorted(restored.weights.values(), key=lambda w: w.path_id)] == [
        w.to_dict() for w in sorted(store.weights.values(), key=lambda w: w.path_id)
    ]
    assert [e.to_dict() for e in restored.update_events] == [e.to_dict() for e in store.update_events]
    # The restored store continues the event-id sequence monotonically.
    next_event = restored.propose(weight.path_id, 0.55, reason="post-restore", feedback_source="read_hit")
    assert next_event.event_id == "weight-event-000005"


def test_stage151_path_ids_are_deterministic_and_directional() -> None:
    forward = MemoryLink("a", "b", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    backward = MemoryLink("b", "a", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    assert path_id_for_link(forward) != path_id_for_link(backward)
    assert path_id_for_link(forward) == path_id_for_link(MemoryLink("a", "b", MemoryLinkType.TEMPORAL, 9.9, 5.0))


def test_stage151_fail_closed_on_unknown_path_and_bad_bounds() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)

    # Fail closed: operating on an unregistered path is rejected, never silent.
    with pytest.raises(KeyError):
        store.propose("link:missing:missing:temporal", 0.5, reason="x", feedback_source="read_hit")
    with pytest.raises(KeyError):
        store.approve("link:missing:missing:temporal", 0.5, reason="x", feedback_source="read_hit")
    with pytest.raises(KeyError):
        store.decay("link:missing:missing:temporal", decay_rate=0.1)
    with pytest.raises(KeyError):
        store.reject("link:missing:missing:temporal", 0.5, reason="x", feedback_source="read_hit")

    weight = store.register_link(_link(weight=0.5))
    with pytest.raises(ValueError):
        store.decay(weight.path_id, decay_rate=1.5)
    with pytest.raises(ValueError):
        store.decay(weight.path_id, decay_rate=-0.1)
    with pytest.raises(ValueError):
        store.clip(weight.path_id, lo=1.0, hi=0.0)


def test_stage151_event_types_round_trip_through_dict() -> None:
    store = HelixWeightStore(now_fn=lambda: 1.0)
    weight = store.register_link(_link(weight=0.5))
    store.propose(weight.path_id, 0.9, reason="candidate", feedback_source="read_hit")
    store.approve(weight.path_id, 0.8, reason="approved", feedback_source="skill_success")
    store.decay(weight.path_id, decay_rate=0.2)

    for event in store.update_events:
        restored = WeightUpdateEvent.from_dict(event.to_dict())
        assert restored == event
        assert isinstance(restored.action, WeightUpdateAction)

    restored_weight = PathWeight.from_dict(weight.to_dict())
    # ``weight`` is the frozen initial snapshot (v0); the live store entry has
    # since been replaced by approve/decay. Serialization must reproduce the
    # snapshot exactly, not the mutated current value.
    assert restored_weight == weight
