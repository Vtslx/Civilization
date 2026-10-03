from __future__ import annotations

import json

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage151_helix_path_weight_model import (
    HelixWeightStore,
    path_id_for_link,
)
from experiments.civilization_transformer_qwen3.analysis.stage152_helix_feedback_signal import (
    FeedbackKind,
    FeedbackSignal,
    HIT_MAGNITUDE,
    REJECTED_MAGNITUDE,
)
from experiments.civilization_transformer_qwen3.analysis.stage153_helix_weight_update_rule import (
    HelixUpdateConfig,
    HelixWeightUpdateRule,
    WeightUpdateProposal,
    run_stage153_helix_weight_update_rule_smoke,
)
from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import (
    MemoryLink,
    MemoryLinkType,
)


def _signal(path_id: str, magnitude: float, signal_id: str = "signal-000001") -> FeedbackSignal:
    return FeedbackSignal(
        signal_id=signal_id,
        source_path_id=path_id,
        kind=FeedbackKind.HIT,
        magnitude=magnitude,
        evidence_trace_id="trace-000001",
        created_at=1.0,
        details={},
    )


def _store_with_link(weight: float = 1.0) -> tuple[HelixWeightStore, str]:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    link = MemoryLink("episodic-000001", "episodic-000002", MemoryLinkType.TEMPORAL, weight, 0.0)
    pw = store.register_link(link)
    return store, pw.path_id


def test_stage153_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage153_helix_weight_update_rule_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage153_helix_weight_update_rule"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "weight_update_proposals.json").exists()


def test_stage153_propose_is_pure() -> None:
    store, path_id = _store_with_link(weight=1.0)
    signals = [_signal(path_id, HIT_MAGNITUDE)]
    snapshot = store.to_dict()

    HelixWeightUpdateRule().propose(store, signals)

    assert store.to_dict() == snapshot  # store untouched by propose


def test_stage153_propose_is_idempotent() -> None:
    store, path_id = _store_with_link(weight=1.0)
    signals = [_signal(path_id, HIT_MAGNITUDE, "s1"), _signal(path_id, -0.4, "s2")]
    rule = HelixWeightUpdateRule()

    first = rule.propose(store, signals)
    second = rule.propose(store, signals)

    assert [p.to_dict() for p in second] == [p.to_dict() for p in first]


def test_stage153_feedback_update_formula() -> None:
    store, path_id = _store_with_link(weight=1.0)
    rule_obj = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.5, decay_rate=0.1, w_min=0.0, w_max=2.0))
    signals = [_signal(path_id, 0.8, "s1"), _signal(path_id, -0.2, "s2")]  # sum = 0.6

    proposals = rule_obj.propose(store, signals)
    proposal = next(p for p in proposals if p.path_id == path_id)

    # candidate = 1.0 + 0.5 * 0.6 = 1.3, within bounds -> not clipped
    assert proposal.new_value == pytest.approx(1.3)
    assert proposal.clipped is False
    assert proposal.raw_candidate == pytest.approx(1.3)
    assert proposal.reason == "feedback_update"
    assert proposal.feedback_source == "stage153_rule"
    assert proposal.signal_ids == ("s1", "s2")


def test_stage153_clipping_at_upper_bound() -> None:
    store, path_id = _store_with_link(weight=1.0)
    # Boost near the ceiling so the update overshoots w_max.
    store.approve(path_id, 1.9, reason="boost", feedback_source="test")
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.5, decay_rate=0.1, w_min=0.0, w_max=2.0))
    signals = [_signal(path_id, 0.8)]  # candidate = 1.9 + 0.5*0.8 = 2.3 -> clip to 2.0

    proposals = rule.propose(store, signals)
    proposal = next(p for p in proposals if p.path_id == path_id)

    assert proposal.new_value == pytest.approx(2.0)
    assert proposal.clipped is True
    assert proposal.raw_candidate == pytest.approx(2.3)


def test_stage153_clipping_at_lower_bound() -> None:
    store, path_id = _store_with_link(weight=0.2)
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=1.0, decay_rate=0.1, w_min=0.0, w_max=2.0))
    signals = [_signal(path_id, -0.8)]  # candidate = 0.2 + 1.0*(-0.8) = -0.6 -> clip to 0.0

    proposals = rule.propose(store, signals)
    proposal = next(p for p in proposals if p.path_id == path_id)

    assert proposal.new_value == pytest.approx(0.0)
    assert proposal.clipped is True
    assert proposal.raw_candidate == pytest.approx(-0.6)


def test_stage153_unused_decay_moves_toward_baseline() -> None:
    store, path_id = _store_with_link(weight=0.5)
    store.approve(path_id, 0.9, reason="boost", feedback_source="test")  # above baseline 0.5
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.3, decay_rate=0.2, w_min=0.0, w_max=2.0))

    proposals = rule.propose(store, signals=[])  # no signals -> unused decay
    proposal = next(p for p in proposals if p.path_id == path_id)

    # candidate = 0.9 + 0.2*(0.5-0.9) = 0.82, between baseline 0.5 and old 0.9
    assert proposal.new_value == pytest.approx(0.82)
    assert proposal.reason == "decay"
    assert proposal.signal_ids == ()
    assert 0.5 < proposal.new_value < 0.9  # moved toward baseline without crossing


def test_stage153_decay_converges_to_baseline_under_repeated_application() -> None:
    store, path_id = _store_with_link(weight=0.5)
    store.approve(path_id, 1.5, reason="boost", feedback_source="test")
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.3, decay_rate=0.5, w_min=0.0, w_max=2.0))

    # Repeated decay-only rounds monotonically approach the baseline 0.5.
    values = [store.weights[path_id].value]
    for _ in range(8):
        rule.apply(store, signals=[])
        values.append(store.weights[path_id].value)

    assert all(b <= a for a, b in zip(values, values[1:]))  # monotonic decreasing toward 0.5
    assert abs(values[-1] - 0.5) < 0.01  # converged near baseline


def test_stage153_apply_commits_proposals_to_store() -> None:
    store, path_id = _store_with_link(weight=1.0)
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.3, decay_rate=0.1, w_min=0.0, w_max=2.0))
    signals = [_signal(path_id, HIT_MAGNITUDE)]  # candidate = 1.0 + 0.3*0.5 = 1.15

    proposals = rule.apply(store, signals)

    assert store.weights[path_id].value == pytest.approx(1.15)
    assert store.weights[path_id].version == 1  # register(0) -> apply approve(1)
    assert proposals[0].new_value == pytest.approx(1.15)


def test_stage153_apply_unused_path_emits_decay_event() -> None:
    store, path_id = _store_with_link(weight=0.5)
    store.approve(path_id, 1.0, reason="boost", feedback_source="test")
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=0.3, decay_rate=0.5, w_min=0.0, w_max=2.0))

    before_events = len(store.update_events)
    rule.apply(store, signals=[])
    after_events = len(store.update_events)

    # The unused path was updated via store.decay -> a DECAY event was appended.
    from experiments.civilization_transformer_qwen3.analysis.stage151_helix_path_weight_model import (
        WeightUpdateAction,
    )

    new_events = store.update_events[before_events:]
    assert any(e.action is WeightUpdateAction.DECAY for e in new_events)
    assert after_events > before_events


def test_stage153_fail_closed_on_unregistered_signal_path() -> None:
    store, _ = _store_with_link(weight=1.0)
    # A signal for a path that was never registered in the weight store.
    bogus_signal = _signal("link:ghost-000001:ghost-000002:temporal", 0.5)

    with pytest.raises(KeyError):
        HelixWeightUpdateRule().propose(store, [bogus_signal])


def test_stage153_query_level_signals_are_skipped() -> None:
    store, path_id = _store_with_link(weight=1.0)
    miss = FeedbackSignal(
        signal_id="s-miss",
        source_path_id="",  # query-level read miss
        kind=FeedbackKind.MISS,
        magnitude=-0.3,
        evidence_trace_id="trace-000010",
        created_at=1.0,
        details={},
    )

    proposals = HelixWeightUpdateRule().propose(store, [miss])

    # The miss carries no link target, so the only path decays (unused).
    assert len(proposals) == 1
    assert proposals[0].path_id == path_id
    assert proposals[0].reason == "decay"
    assert proposals[0].signal_ids == ()


def test_stage153_deterministic_order_by_path_id() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    links = [
        MemoryLink("c-3", "c-4", MemoryLinkType.TEMPORAL, 1.0, 0.0),
        MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0),
        MemoryLink("c-5", "c-6", MemoryLinkType.TEMPORAL, 1.0, 0.0),
    ]
    for link in links:
        store.register_link(link)

    proposals = HelixWeightUpdateRule().propose(store, signals=[])

    assert [p.path_id for p in proposals] == sorted(path_id_for_link(lk) for lk in links)


def test_stage153_proposal_round_trip_serialization() -> None:
    proposal = WeightUpdateProposal(
        path_id="link:a:b:temporal",
        old_value=1.0,
        new_value=1.3,
        reason="feedback_update",
        feedback_source="stage153_rule",
        signal_ids=("s1", "s2"),
        raw_candidate=1.3,
        clipped=False,
        baseline=1.0,
    )
    restored = WeightUpdateProposal.from_dict(json.loads(json.dumps(proposal.to_dict(), ensure_ascii=False)))
    assert restored == proposal


def test_stage153_bounded_across_random_signal_mixes() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    link = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    store.register_link(link)
    path_id = path_id_for_link(link)

    # Extreme magnitudes (still within Stage152's [-1, 1]) with a large lr.
    rule = HelixWeightUpdateRule(HelixUpdateConfig(lr=5.0, decay_rate=0.9, w_min=0.0, w_max=2.0))
    signals = [_signal(path_id, 1.0, "s1"), _signal(path_id, 1.0, "s2"), _signal(path_id, 1.0, "s3")]

    proposals = rule.propose(store, signals)

    assert all(0.0 <= p.new_value <= 2.0 for p in proposals)
    assert proposals[0].clipped is True  # 1.0 + 5.0*3.0 = 16.0 -> clipped to 2.0
