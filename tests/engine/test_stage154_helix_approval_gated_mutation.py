from __future__ import annotations

import json

import pytest

from civilization.engine.stages.stage151_helix_path_weight_model import (
    HelixWeightStore,
    MemoryLink,
    WeightUpdateAction,
    path_id_for_link,
)
from civilization.engine.stages.stage153_helix_weight_update_rule import (
    WeightUpdateProposal,
)
from civilization.engine.stages.stage154_helix_approval_gated_mutation import (
    ApprovalDecision,
    ApprovalRecord,
    ApproveAllPolicy,
    HelixApprovalGate,
    MaxDeltaPolicy,
    candidate_id_for,
    run_stage154_helix_approval_gated_mutation_smoke,
)
from civilization.engine.stages.stage73_orion_memory_kernel import (
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


def _store_with_link(weight: float = 1.0) -> tuple[HelixWeightStore, str]:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    link = MemoryLink("episodic-000001", "episodic-000002", MemoryLinkType.TEMPORAL, weight, 0.0)
    pw = store.register_link(link)
    return store, pw.path_id


def test_stage154_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage154_helix_approval_gated_mutation_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage154_helix_approval_gated_mutation"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "approval_records.json").exists()


def test_stage154_approved_candidate_is_committed() -> None:
    store, path_id = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 1.0)
    proposals = [_proposal(path_id, old=1.0, new=1.3)]

    records = gate.evaluate(store, proposals)

    assert records[0].decision is ApprovalDecision.APPROVED
    assert store.weights[path_id].value == pytest.approx(1.3)
    assert store.weights[path_id].version == 1
    # An APPROVE event was appended for the commit.
    assert any(e.action is WeightUpdateAction.APPROVE and e.path_id == path_id for e in store.update_events)


def test_stage154_rejected_candidate_not_committed() -> None:
    store, path_id = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.1), now_fn=lambda: 1.0)
    proposals = [_proposal(path_id, old=1.0, new=1.5)]  # delta 0.5 > 0.1 -> rejected

    records = gate.evaluate(store, proposals)

    assert records[0].decision is ApprovalDecision.REJECTED
    assert store.weights[path_id].value == pytest.approx(1.0)  # unchanged
    assert store.weights[path_id].version == 0  # no version bump


def test_stage154_rejected_candidate_queryable_in_trace() -> None:
    store, path_id = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.1), now_fn=lambda: 1.0)
    gate.evaluate(store, [_proposal(path_id, old=1.0, new=1.5)])

    reject_events = [e for e in store.update_events if e.action is WeightUpdateAction.REJECT]
    assert len(reject_events) == 1
    assert reject_events[0].path_id == path_id
    assert reject_events[0].new_value == pytest.approx(1.5)  # rejected candidate preserved


def test_stage154_preserve_both_on_reject() -> None:
    store, path_id = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.1), now_fn=lambda: 1.0)
    records = gate.evaluate(store, [_proposal(path_id, old=1.0, new=1.5)])

    # The rejected candidate value is in the record/event; the original survives in the store.
    assert records[0].new_value == pytest.approx(1.5)
    assert store.weights[path_id].value == pytest.approx(1.0)
    reject_events = [e for e in store.update_events if e.action is WeightUpdateAction.REJECT]
    assert reject_events[0].old_value == pytest.approx(1.0)  # original preserved
    assert reject_events[0].new_value == pytest.approx(1.5)  # candidate preserved


def test_stage154_version_monotonic_across_mixed_decisions() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    link_a = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    link_b = MemoryLink("c-3", "c-4", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    pa = store.register_link(link_a).path_id
    pb = store.register_link(link_b).path_id
    versions_before = {pa: store.weights[pa].version, pb: store.weights[pb].version}

    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.2), now_fn=lambda: 1.0)
    gate.evaluate(store, [_proposal(pa, old=1.0, new=1.1), _proposal(pb, old=1.0, new=1.5)])

    # pa approved (delta 0.1) -> version 1; pb rejected (delta 0.5) -> version 0.
    assert store.weights[pa].version == 1
    assert store.weights[pb].version == 0
    assert all(store.weights[pid].version >= versions_before[pid] for pid in (pa, pb))


def test_stage154_idempotent_re_evaluation() -> None:
    store, path_id = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 1.0)
    proposals = [_proposal(path_id, old=1.0, new=1.3)]

    first = gate.evaluate(store, proposals)
    events_after_first = len(store.update_events)
    weights_after_first = {pid: (w.value, w.version) for pid, w in store.weights.items()}

    second = gate.evaluate(store, proposals)

    # Same records, no new events, no store mutation.
    assert [r.to_dict() for r in second] == [r.to_dict() for r in first]
    assert len(store.update_events) == events_after_first
    assert {pid: (w.value, w.version) for pid, w in store.weights.items()} == weights_after_first


def test_stage154_stale_blocks_double_apply_with_fresh_gate() -> None:
    store, path_id = _store_with_link(weight=1.0)
    proposals = [_proposal(path_id, old=1.0, new=1.3)]

    # First gate approves and commits (value 1.0 -> 1.3).
    HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 1.0).evaluate(store, proposals)
    assert store.weights[path_id].value == pytest.approx(1.3)

    # A fresh gate re-evaluates the SAME proposals; the store value (1.3) no
    # longer matches old_value (1.0), so the candidate is STALE and not re-applied.
    fresh = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 2.0)
    records = fresh.evaluate(store, proposals)

    assert records[0].decision is ApprovalDecision.STALE
    assert store.weights[path_id].value == pytest.approx(1.3)  # no double-apply
    assert store.weights[path_id].version == 1  # no extra version bump


def test_stage154_approve_all_policy() -> None:
    store, path_id = _store_with_link(weight=0.5)
    gate = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 0.0)

    records = gate.evaluate(store, [_proposal(path_id, old=0.5, new=1.9)])

    assert records[0].decision is ApprovalDecision.APPROVED
    assert records[0].approver == "approve_all"
    assert store.weights[path_id].value == pytest.approx(1.9)


def test_stage154_max_delta_policy_boundary() -> None:
    policy = MaxDeltaPolicy(max_delta=0.2)
    # delta exactly 0.2 is within bounds (not > max_delta) -> approved.
    approved, _ = policy.decide(_proposal("p", old=1.0, new=1.2))
    assert approved is ApprovalDecision.APPROVED
    # delta 0.2000001 exceeds -> rejected.
    rejected, reason = policy.decide(_proposal("p", old=1.0, new=1.2000001))
    assert rejected is ApprovalDecision.REJECTED
    assert "exceeds max_delta" in reason


def test_stage154_candidate_id_is_deterministic() -> None:
    p1 = _proposal("link:a:b:temporal", old=1.0, new=1.3)
    p2 = _proposal("link:a:b:temporal", old=1.0, new=1.3)
    p3 = _proposal("link:a:b:temporal", old=1.0, new=1.4)
    assert candidate_id_for(p1) == candidate_id_for(p2)
    assert candidate_id_for(p1) != candidate_id_for(p3)


def test_stage154_fail_closed_on_unregistered_path() -> None:
    store, _ = _store_with_link(weight=1.0)
    gate = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: 0.0)
    bogus = _proposal("link:ghost:ghost:temporal", old=1.0, new=1.3)

    with pytest.raises(KeyError):
        gate.evaluate(store, [bogus])


def test_stage154_approval_record_round_trip() -> None:
    record = ApprovalRecord(
        candidate_id="candidate:p:1.0:1.3:feedback_update",
        path_id="link:a:b:temporal",
        decision=ApprovalDecision.APPROVED,
        old_value=1.0,
        new_value=1.3,
        proposal_reason="feedback_update",
        approver="max_delta",
        decision_reason="delta 0.3 within max_delta 0.5",
        decided_at=6000.0,
    )
    restored = ApprovalRecord.from_dict(json.loads(json.dumps(record.to_dict(), ensure_ascii=False)))
    assert restored == record
    assert isinstance(restored.decision, ApprovalDecision)


def test_stage154_mixed_approve_and_reject_in_one_batch() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    links = [
        MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0),
        MemoryLink("c-3", "c-4", MemoryLinkType.CONFLICT, 1.0, 0.0),
        MemoryLink("c-5", "c-6", MemoryLinkType.PROCEDURE, 0.5, 0.0),
    ]
    pids = [store.register_link(lk).path_id for lk in links]
    proposals = [
        _proposal(pids[0], old=1.0, new=1.1),  # delta 0.1 -> approved
        _proposal(pids[1], old=1.0, new=0.5),  # delta 0.5 -> rejected
        _proposal(pids[2], old=0.5, new=0.6),  # delta 0.1 -> approved
    ]
    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.2), now_fn=lambda: 1.0)

    records = gate.evaluate(store, proposals)

    decisions = {r.path_id: r.decision for r in records}
    assert decisions[pids[0]] is ApprovalDecision.APPROVED
    assert decisions[pids[1]] is ApprovalDecision.REJECTED
    assert decisions[pids[2]] is ApprovalDecision.APPROVED
    assert store.weights[pids[0]].value == pytest.approx(1.1)
    assert store.weights[pids[1]].value == pytest.approx(1.0)  # rejected, unchanged
    assert store.weights[pids[2]].value == pytest.approx(0.6)
