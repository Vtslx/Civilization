from __future__ import annotations

import json

import pytest

from civilization.engine.stages.stage151_helix_path_weight_model import (
    HelixWeightStore,
    path_id_for_link,
)
from civilization.engine.stages.stage153_helix_weight_update_rule import (
    WeightUpdateProposal,
)
from civilization.engine.stages.stage154_helix_approval_gated_mutation import (
    candidate_id_for,
)
from civilization.engine.stages.stage156_helix_weight_conflict_guard import (
    ConflictGuardResult,
    ConflictReason,
    HelixWeightConflictGuard,
    WeightConflictRecord,
    run_stage156_helix_weight_conflict_guard_smoke,
)
from civilization.engine.stages.stage73_orion_memory_kernel import (
    MemoryLink,
    MemoryLinkType,
)


def _proposal(path_id: str, old: float, new: float, reason: str = "feedback_update") -> WeightUpdateProposal:
    return WeightUpdateProposal(
        path_id=path_id,
        old_value=old,
        new_value=new,
        reason=reason,
        feedback_source="stage153_rule",
        signal_ids=("signal-000001",),
        raw_candidate=new,
        clipped=False,
        baseline=old,
    )


def _store_with_links(*weights: float) -> tuple[HelixWeightStore, list[str]]:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    pids = []
    for i, w in enumerate(weights):
        link = MemoryLink(f"c-{2*i+1}", f"c-{2*i+2}", MemoryLinkType.TEMPORAL, w, 0.0)
        pids.append(store.register_link(link).path_id)
    return store, pids


def test_stage156_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage156_helix_weight_conflict_guard_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage156_helix_weight_conflict_guard"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "conflict_guard_result.json").exists()


def test_stage156_divergent_candidates_preserve_both() -> None:
    store, [pid] = _store_with_links(1.0)
    proposals = [_proposal(pid, 1.0, 1.4), _proposal(pid, 1.0, 0.6)]  # opposing directions

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 0
    assert len(result.review_queue) == 2  # both preserved
    record = next(r for r in result.conflict_records if r.reason is ConflictReason.DIVERGENT)
    assert len(record.candidate_ids) == 2
    assert record.deltas == pytest.approx((0.4, -0.4))


def test_stage156_oscillating_candidate_detected() -> None:
    store, [pid] = _store_with_links(0.5)
    store.approve(pid, 1.0, reason="prior increase", feedback_source="t")  # last direction: +
    proposals = [_proposal(pid, 1.0, 0.7)]  # now wants decrease -> oscillating

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 0
    assert len(result.review_queue) == 1
    record = next(r for r in result.conflict_records if r.reason is ConflictReason.OSCILLATING)
    assert record.last_committed_direction == 1
    assert record.deltas == pytest.approx((-0.3,))


def test_stage156_clean_candidate_passes_through() -> None:
    store, [pid] = _store_with_links(1.0)  # no history
    proposals = [_proposal(pid, 1.0, 1.2)]

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 1
    assert result.clean_candidates[0].path_id == pid
    assert len(result.review_queue) == 0
    assert len(result.conflict_records) == 0


def test_stage156_same_direction_as_history_is_clean() -> None:
    store, [pid] = _store_with_links(0.5)
    store.approve(pid, 1.0, reason="prior increase", feedback_source="t")  # last: +
    proposals = [_proposal(pid, 1.0, 1.3)]  # also increase -> not oscillating

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 1
    assert len(result.review_queue) == 0


def test_stage156_no_candidate_lost() -> None:
    store, [pid_a, pid_b, pid_c] = _store_with_links(1.0, 1.0, 0.5)
    store.approve(pid_c, 1.0, reason="up", feedback_source="t")
    proposals = [
        _proposal(pid_a, 1.0, 1.2),       # clean
        _proposal(pid_b, 1.0, 1.4),       # divergent pair
        _proposal(pid_b, 1.0, 0.6),
        _proposal(pid_c, 1.0, 0.7),       # oscillating
    ]

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    input_ids = sorted(candidate_id_for(c) for c in proposals)
    output_ids = sorted(candidate_id_for(c) for c in [*result.clean_candidates, *result.review_queue])
    assert input_ids == output_ids  # every candidate is in clean or review, none dropped


def test_stage156_read_only() -> None:
    store, [pid] = _store_with_links(1.0)
    proposals = [_proposal(pid, 1.0, 1.4), _proposal(pid, 1.0, 0.6)]
    snapshot = store.to_dict()

    HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert store.to_dict() == snapshot  # store untouched


def test_stage156_deterministic() -> None:
    store, [pid_a, pid_b] = _store_with_links(1.0, 0.5)
    store.approve(pid_b, 1.0, reason="up", feedback_source="t")
    proposals = [_proposal(pid_a, 1.0, 1.2), _proposal(pid_b, 1.0, 0.7)]
    guard = HelixWeightConflictGuard(now_fn=lambda: 0.0)

    first = guard.evaluate(store, proposals)
    second = guard.evaluate(store, proposals)

    assert second.to_dict() == first.to_dict()


def test_stage156_preserve_both_distinct_in_review() -> None:
    store, [pid] = _store_with_links(1.0)
    proposals = [_proposal(pid, 1.0, 1.4), _proposal(pid, 1.0, 0.6)]

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    divergent = [c for c in result.review_queue if c.path_id == pid]
    assert len(divergent) == 2
    assert {c.new_value for c in divergent} == {1.4, 0.6}  # both distinct, not merged


def test_stage156_no_history_means_no_oscillation() -> None:
    store, [pid] = _store_with_links(1.0)  # registered only, no committed update
    proposals = [_proposal(pid, 1.0, 0.6)]  # decrease, but no history to oppose

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 1
    assert len(result.conflict_records) == 0


def test_stage156_init_event_does_not_count_as_direction() -> None:
    """Registration emits an APPROVE(init) with delta 0; it must not set a direction."""
    store, [pid] = _store_with_links(1.0)  # register -> APPROVE(init, delta 0)
    # A decrease candidate: if init counted as direction 0, behavior should still
    # treat "no non-zero direction" as no oscillation.
    proposals = [_proposal(pid, 1.0, 0.6)]

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    assert len(result.clean_candidates) == 1  # no real history -> clean


def test_stage156_conflict_record_round_trip() -> None:
    record = WeightConflictRecord(
        conflict_id="conflict-000001",
        path_id="link:a:b:temporal",
        reason=ConflictReason.DIVERGENT,
        candidate_ids=("candidate:x:1:1.4:feedback_update", "candidate:x:1:0.6:feedback_update"),
        deltas=(0.4, -0.4),
        last_committed_direction=None,
        detected_at=8000.0,
    )
    restored = WeightConflictRecord.from_dict(json.loads(json.dumps(record.to_dict(), ensure_ascii=False)))
    assert restored == record
    assert isinstance(restored.reason, ConflictReason)


def test_stage156_mixed_batch_routes_correctly() -> None:
    store, [pid_clean, pid_div, pid_osc] = _store_with_links(1.0, 1.0, 0.5)
    store.approve(pid_osc, 1.0, reason="up", feedback_source="t")
    proposals = [
        _proposal(pid_clean, 1.0, 1.2),
        _proposal(pid_div, 1.0, 1.4),
        _proposal(pid_div, 1.0, 0.6),
        _proposal(pid_osc, 1.0, 0.7),
    ]

    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, proposals)

    clean_paths = {c.path_id for c in result.clean_candidates}
    review_paths = {c.path_id for c in result.review_queue}
    assert clean_paths == {pid_clean}
    assert review_paths == {pid_div, pid_osc}
    reasons = {r.reason for r in result.conflict_records}
    assert reasons == {ConflictReason.DIVERGENT, ConflictReason.OSCILLATING}


def test_stage156_conflict_record_auditable_with_candidate_ids() -> None:
    store, [pid] = _store_with_links(1.0)
    c1 = _proposal(pid, 1.0, 1.4)
    c2 = _proposal(pid, 1.0, 0.6)
    result = HelixWeightConflictGuard(now_fn=lambda: 0.0).evaluate(store, [c1, c2])

    record = result.conflict_records[0]
    assert candidate_id_for(c1) in record.candidate_ids
    assert candidate_id_for(c2) in record.candidate_ids
    # The record is queryable by path_id and reason.
    assert record.path_id == pid
    assert record.reason is ConflictReason.DIVERGENT
