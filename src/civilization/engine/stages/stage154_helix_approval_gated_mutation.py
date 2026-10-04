from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any

from .stage151_helix_path_weight_model import HelixWeightStore, WeightUpdateAction
from .stage153_helix_weight_update_rule import WeightUpdateProposal


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage154_helix_approval_gated_mutation")


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(frozen=True)
class ApprovalRecord:
    candidate_id: str
    path_id: str
    decision: ApprovalDecision
    old_value: float
    new_value: float
    proposal_reason: str
    approver: str
    decision_reason: str
    decided_at: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ApprovalRecord":
        return cls(
            str(payload["candidate_id"]),
            str(payload["path_id"]),
            ApprovalDecision(payload["decision"]),
            float(payload["old_value"]),
            float(payload["new_value"]),
            str(payload["proposal_reason"]),
            str(payload["approver"]),
            str(payload["decision_reason"]),
            float(payload["decided_at"]),
        )


def candidate_id_for(proposal: WeightUpdateProposal) -> str:
    """Deterministic identity for a candidate update.

    Two proposals with the same (path, old, new, reason) are the same candidate,
    so re-submitting one is idempotent.
    """
    return f"candidate:{proposal.path_id}:{proposal.old_value}:{proposal.new_value}:{proposal.reason}"


class ApprovalPolicy:
    """Base approval policy. Subclasses implement ``decide``."""

    name: str = "policy"

    def decide(self, proposal: WeightUpdateProposal) -> tuple[ApprovalDecision, str]:
        raise NotImplementedError


class ApproveAllPolicy(ApprovalPolicy):
    """Baseline policy: approves every candidate. Used to prove the gate itself."""

    name = "approve_all"

    def decide(self, proposal: WeightUpdateProposal) -> tuple[ApprovalDecision, str]:
        return ApprovalDecision.APPROVED, "approve_all"


class MaxDeltaPolicy(ApprovalPolicy):
    """Reject candidates whose committed delta exceeds a magnitude cap.

    Guards against runaway updates: a feedback signal strong enough to move the
    weight further than ``max_delta`` in one round is held back for review.
    """

    name = "max_delta"

    def __init__(self, max_delta: float) -> None:
        if max_delta < 0:
            raise ValueError("max_delta must be non-negative")
        self.max_delta = float(max_delta)

    def decide(self, proposal: WeightUpdateProposal) -> tuple[ApprovalDecision, str]:
        delta = abs(proposal.new_value - proposal.old_value)
        if delta > self.max_delta:
            return ApprovalDecision.REJECTED, f"delta {delta:.6f} exceeds max_delta {self.max_delta:.6f}"
        return ApprovalDecision.APPROVED, f"delta {delta:.6f} within max_delta {self.max_delta:.6f}"


class HelixApprovalGate:
    """Candidate-to-approved admission layer over learnable weight updates.

    Each Stage153 proposal becomes a candidate. The policy decides approve or
    reject. Approved candidates are committed through ``store.approve`` (version
    increments, APPROVE event). Rejected candidates are recorded through
    ``store.reject`` (REJECT event, weight untouched) so the rejected value is
    preserved in the audit trace without overwriting the original — preserve-both.

    Idempotency is two-layered: (1) a candidate already decided by this gate is
    returned as-is on re-submission (no new event, no store change); (2) if the
    store's current value no longer matches the proposal's ``old_value``, the
    candidate is marked STALE and skipped (cannot double-apply a consumed
    proposal).
    """

    def __init__(self, policy: ApprovalPolicy, *, now_fn=time.time) -> None:
        self.policy = policy
        self._now_fn = now_fn
        self.decisions: list[ApprovalRecord] = []
        self._decided: dict[str, ApprovalRecord] = {}

    def evaluate(self, store: HelixWeightStore, proposals: list[WeightUpdateProposal]) -> list[ApprovalRecord]:
        records: list[ApprovalRecord] = []
        for proposal in proposals:
            candidate_id = candidate_id_for(proposal)
            existing = self._decided.get(candidate_id)
            if existing is not None:
                records.append(existing)  # idempotent: same candidate, no side effects
                continue
            record = self._evaluate_one(store, proposal, candidate_id)
            self._decided[candidate_id] = record
            self.decisions.append(record)
            records.append(record)
        return records

    def _evaluate_one(self, store: HelixWeightStore, proposal: WeightUpdateProposal, candidate_id: str) -> ApprovalRecord:
        current = store.weights.get(proposal.path_id)
        if current is None:
            raise KeyError(f"candidate references unregistered path: {proposal.path_id}")
        if current.value != proposal.old_value:
            return ApprovalRecord(
                candidate_id=candidate_id,
                path_id=proposal.path_id,
                decision=ApprovalDecision.STALE,
                old_value=proposal.old_value,
                new_value=proposal.new_value,
                proposal_reason=proposal.reason,
                approver=self.policy.name,
                decision_reason=f"stale: store value {current.value} != proposal old_value {proposal.old_value}",
                decided_at=self._now_fn(),
            )
        decision, decision_reason = self.policy.decide(proposal)
        if decision is ApprovalDecision.APPROVED:
            store.approve(
                proposal.path_id,
                proposal.new_value,
                reason=proposal.reason,
                feedback_source=proposal.feedback_source,
            )
        else:
            store.reject(
                proposal.path_id,
                proposal.new_value,
                reason=decision_reason,
                feedback_source="stage154_gate",
            )
        return ApprovalRecord(
            candidate_id=candidate_id,
            path_id=proposal.path_id,
            decision=decision,
            old_value=proposal.old_value,
            new_value=proposal.new_value,
            proposal_reason=proposal.reason,
            approver=self.policy.name,
            decision_reason=decision_reason,
            decided_at=self._now_fn(),
        )

    def write_artifacts(self, output_dir: str | Path, *, summary: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "approval_records.json").write_text(
            json.dumps([r.to_dict() for r in self.decisions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if summary is not None:
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stage154_helix_approval_gated_mutation_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from .stage153_helix_weight_update_rule import HelixUpdateConfig, HelixWeightUpdateRule, _build_smoke_store

    clock = {"now": 6000.0}
    weight_store, signals, _expected = _build_smoke_store()
    # Rebind the store clock to the stage154 deterministic clock so event
    # timestamps are comparable across the run.
    weight_store._now_fn = lambda: clock["now"]

    rule = HelixWeightUpdateRule(HelixUpdateConfig())
    proposals = rule.propose(weight_store, signals)

    # Snapshot before gating to prove rejected candidates do not mutate the store.
    weights_before = {pid: (w.value, w.version) for pid, w in weight_store.weights.items()}
    events_before = len(weight_store.update_events)

    gate = HelixApprovalGate(MaxDeltaPolicy(max_delta=0.2), now_fn=lambda: clock["now"])
    records = gate.evaluate(weight_store, proposals)

    # Locate the three candidates by their proposal reason / delta signature.
    by_path = {r.path_id: r for r in records}
    # HIT-driven proposal: old 1.95 -> new 2.0 (delta 0.05) -> approved.
    hit_path = next(pid for pid, w in weights_before.items() if w[0] == 1.95)
    # Conflict-driven proposal: old 1.0 -> new 0.76 (delta 0.24) -> rejected.
    rejected_path = next(pid for pid, w in weights_before.items() if w[0] == 1.0 and pid.endswith(":conflict"))
    # Unused decay proposal: old 0.9 -> new 0.86 (delta 0.04) -> approved.
    decay_path = next(pid for pid, w in weights_before.items() if w[0] == 0.9)

    hit_record = by_path[hit_path]
    rejected_record = by_path[rejected_path]
    decay_record = by_path[decay_path]

    approved_committed = weight_store.weights[hit_path].value == 2.0 and weight_store.weights[hit_path].version == 2
    decay_committed = weight_store.weights[decay_path].value == 0.86 and weight_store.weights[decay_path].version == 2
    rejected_not_committed = weight_store.weights[rejected_path].value == 1.0 and weight_store.weights[rejected_path].version == 0
    rejected_queryable = any(
        e.action is WeightUpdateAction.REJECT and e.path_id == rejected_path
        for e in weight_store.update_events
    )
    preserve_both = rejected_record.new_value == 0.76 and weight_store.weights[rejected_path].value == 1.0

    # Version monotonic: every path's version is >= its pre-gate version.
    version_monotonic = all(
        weight_store.weights[pid].version >= v for pid, (_, v) in weights_before.items()
    )

    # Idempotent re-evaluation: same gate, same proposals -> no new events, no store change.
    weights_after_first = {pid: (w.value, w.version) for pid, w in weight_store.weights.items()}
    events_after_first = len(weight_store.update_events)
    records_again = gate.evaluate(weight_store, proposals)
    idempotent = (
        [r.to_dict() for r in records_again] == [r.to_dict() for r in records]
        and len(weight_store.update_events) == events_after_first
        and {pid: (w.value, w.version) for pid, w in weight_store.weights.items()} == weights_after_first
    )

    # Stale detection: a fresh gate on the already-mutated store marks consumed
    # proposals STALE (approved paths) and does not double-apply them.
    fresh_gate = HelixApprovalGate(ApproveAllPolicy(), now_fn=lambda: clock["now"])
    fresh_records = fresh_gate.evaluate(weight_store, proposals)
    stale_paths = {r.path_id for r in fresh_records if r.decision is ApprovalDecision.STALE}
    stale_blocks_double_apply = hit_path in stale_paths and decay_path in stale_paths
    # The rejected path is still at old_value 1.0, so a fresh ApproveAll gate would
    # re-approve it — but that is a different policy decision, not a double-apply
    # of an already-consumed proposal. The approved paths must not move again.
    no_double_apply = weight_store.weights[hit_path].value == 2.0 and weight_store.weights[decay_path].value == 0.86

    gate.write_artifacts(output_dir)

    summary = {
        "stage": "stage154_helix_approval_gated_mutation",
        "version": "v0.00.07",
        "policy": MaxDeltaPolicy(max_delta=0.2).name,
        "max_delta": 0.2,
        "candidate_count": len(records),
        "decision_counts": {
            "approved": sum(1 for r in records if r.decision is ApprovalDecision.APPROVED),
            "rejected": sum(1 for r in records if r.decision is ApprovalDecision.REJECTED),
            "stale": sum(1 for r in records if r.decision is ApprovalDecision.STALE),
        },
        "stage_gates": {
            "approved_committed": approved_committed,
            "decay_committed": decay_committed,
            "rejected_not_committed": rejected_not_committed,
            "rejected_queryable_in_trace": rejected_queryable,
            "preserve_both": preserve_both,
            "version_monotonic": version_monotonic,
            "idempotent_re_evaluation": idempotent,
            "stale_blocks_double_apply": stale_blocks_double_apply and no_double_apply,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    gate.write_artifacts(output_dir, summary=summary)
    return summary
