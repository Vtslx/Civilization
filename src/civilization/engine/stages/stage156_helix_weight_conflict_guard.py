from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any

from .stage151_helix_path_weight_model import HelixWeightStore, WeightUpdateAction
from .stage154_helix_approval_gated_mutation import candidate_id_for
from .stage153_helix_weight_update_rule import WeightUpdateProposal


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage156_helix_weight_conflict_guard")


class ConflictReason(StrEnum):
    DIVERGENT = "divergent"
    OSCILLATING = "oscillating"


@dataclass(frozen=True)
class WeightConflictRecord:
    conflict_id: str
    path_id: str
    reason: ConflictReason
    candidate_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    last_committed_direction: int | None
    detected_at: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason"] = self.reason.value
        payload["candidate_ids"] = list(self.candidate_ids)
        payload["deltas"] = list(self.deltas)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightConflictRecord":
        return cls(
            str(payload["conflict_id"]),
            str(payload["path_id"]),
            ConflictReason(payload["reason"]),
            tuple(payload.get("candidate_ids", [])),
            tuple(payload.get("deltas", [])),
            payload.get("last_committed_direction"),
            float(payload["detected_at"]),
        )


@dataclass(frozen=True)
class ConflictGuardResult:
    clean_candidates: tuple[WeightUpdateProposal, ...]
    review_queue: tuple[WeightUpdateProposal, ...]
    conflict_records: tuple[WeightConflictRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_candidates": [c.to_dict() for c in self.clean_candidates],
            "review_queue": [c.to_dict() for c in self.review_queue],
            "conflict_records": [r.to_dict() for r in self.conflict_records],
        }


def _direction(delta: float) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


class HelixWeightConflictGuard:
    """Detects divergent/oscillating weight candidates and routes them to review.

    A candidate is a Stage153 proposal. The guard groups candidates by path and:

    - **Divergent**: two or more candidates for the same path propose opposing
      directions (one wants to increase, another to decrease). All candidates in
      the group are preserved (preserve-both) and routed to the review queue —
      none are merged or auto-applied.
    - **Oscillating**: a candidate's direction opposes the path's most recent
      non-zero committed update direction. The candidate is preserved and routed
      to review.

    Non-conflicting candidates pass through as ``clean_candidates``, eligible for
    the Stage154 approval gate. The guard is read-only: it never mutates the
    store. This mirrors the Lagoon conflict-guard semantic — conflicting
    evidence is preserved, not overwritten.
    """

    def __init__(self, *, now_fn=time.time) -> None:
        self._now_fn = now_fn

    def evaluate(self, store: HelixWeightStore, proposals: list[WeightUpdateProposal]) -> ConflictGuardResult:
        groups: dict[str, list[WeightUpdateProposal]] = {}
        for proposal in proposals:
            groups.setdefault(proposal.path_id, []).append(proposal)

        clean: list[WeightUpdateProposal] = []
        review: list[WeightUpdateProposal] = []
        records: list[WeightConflictRecord] = []
        counter = 0

        for path_id in sorted(groups.keys()):
            candidates = groups[path_id]
            deltas = tuple(c.new_value - c.old_value for c in candidates)
            directions = {_direction(d) for d in deltas}

            # Divergent: opposing directions among candidates for the same path.
            if 1 in directions and -1 in directions:
                counter += 1
                records.append(
                    WeightConflictRecord(
                        conflict_id=f"conflict-{counter:06d}",
                        path_id=path_id,
                        reason=ConflictReason.DIVERGENT,
                        candidate_ids=tuple(candidate_id_for(c) for c in candidates),
                        deltas=deltas,
                        last_committed_direction=self._last_committed_direction(store, path_id),
                        detected_at=self._now_fn(),
                    )
                )
                review.extend(candidates)
                continue

            # Otherwise check each candidate for oscillation against history.
            last_dir = self._last_committed_direction(store, path_id)
            group_conflicted = False
            for candidate, delta in zip(candidates, deltas):
                cand_dir = _direction(delta)
                if last_dir is not None and cand_dir != 0 and cand_dir != last_dir:
                    counter += 1
                    records.append(
                        WeightConflictRecord(
                            conflict_id=f"conflict-{counter:06d}",
                            path_id=path_id,
                            reason=ConflictReason.OSCILLATING,
                            candidate_ids=(candidate_id_for(candidate),),
                            deltas=(delta,),
                            last_committed_direction=last_dir,
                            detected_at=self._now_fn(),
                        )
                    )
                    review.append(candidate)
                    group_conflicted = True
                else:
                    clean.append(candidate)
            # If any candidate in the group conflicted, keep the group's
            # non-conflicted ones as clean (they can still be approved); only
            # the oscillating ones are held back. This preserves per-candidate
            # granularity.

        return ConflictGuardResult(tuple(clean), tuple(review), tuple(records))

    def _last_committed_direction(self, store: HelixWeightStore, path_id: str) -> int | None:
        """Direction of the most recent non-zero APPROVE for ``path_id``."""
        for event in reversed(store.update_events):
            if event.path_id != path_id:
                continue
            if event.action is not WeightUpdateAction.APPROVE:
                continue
            delta = event.new_value - event.old_value
            direction = _direction(delta)
            if direction != 0:
                return direction
        return None

    def write_artifacts(self, result: ConflictGuardResult, output_dir: str | Path, *, summary: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "conflict_guard_result.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if summary is not None:
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


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


def run_stage156_helix_weight_conflict_guard_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from .stage151_helix_path_weight_model import path_id_for_link
    from .stage73_orion_memory_kernel import MemoryLink, MemoryLinkType, OrionMemoryStore

    clock = {"now": 8000.0}
    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])

    # link_clean: no history -> clean candidate.
    link_clean = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    # link_divergent: two opposing candidates in the same batch.
    link_div = MemoryLink("c-3", "c-4", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    # link_oscillating: last committed was an increase; candidate proposes a decrease.
    link_osc = MemoryLink("c-5", "c-6", MemoryLinkType.TEMPORAL, 0.5, 0.0)

    for link in (link_clean, link_div, link_osc):
        weight_store.register_link(link)

    pid_clean = path_id_for_link(link_clean)
    pid_div = path_id_for_link(link_div)
    pid_osc = path_id_for_link(link_osc)

    # Establish a committed increase on link_osc so a later decrease oscillates.
    clock["now"] = 8001.0
    weight_store.approve(pid_osc, 1.0, reason="prior increase", feedback_source="stage156_smoke")

    proposals = [
        _proposal(pid_clean, old=1.0, new=1.2),          # clean (no history, single direction)
        _proposal(pid_div, old=1.0, new=1.4),            # divergent group: wants increase
        _proposal(pid_div, old=1.0, new=0.6),            # divergent group: wants decrease
        _proposal(pid_osc, old=1.0, new=0.7),            # oscillating: last was +, this is -
    ]

    guard = HelixWeightConflictGuard(now_fn=lambda: clock["now"])

    # Read-only proof: snapshot the store, evaluate, verify untouched.
    snapshot = weight_store.to_dict()
    result = guard.evaluate(weight_store, proposals)
    read_only = weight_store.to_dict() == snapshot

    clean_ids = [candidate_id_for(c) for c in result.clean_candidates]
    review_ids = [candidate_id_for(c) for c in result.review_queue]
    input_ids = [candidate_id_for(c) for c in proposals]
    no_candidate_lost = sorted(clean_ids + review_ids) == sorted(input_ids)

    divergent_record = next((r for r in result.conflict_records if r.reason is ConflictReason.DIVERGENT), None)
    oscillating_record = next((r for r in result.conflict_records if r.reason is ConflictReason.OSCILLATING), None)

    divergent_preserve_both = (
        divergent_record is not None
        and len(divergent_record.candidate_ids) == 2
        and candidate_id_for(_proposal(pid_div, 1.0, 1.4)) in divergent_record.candidate_ids
        and candidate_id_for(_proposal(pid_div, 1.0, 0.6)) in divergent_record.candidate_ids
    )
    divergent_not_in_clean = all(c.path_id != pid_div for c in result.clean_candidates)
    oscillating_detected = (
        oscillating_record is not None
        and oscillating_record.last_committed_direction == 1
        and oscillating_record.path_id == pid_osc
    )
    oscillating_not_in_clean = all(c.path_id != pid_osc for c in result.clean_candidates)
    clean_passes_through = any(c.path_id == pid_clean for c in result.clean_candidates)

    # Deterministic: second evaluation identical.
    result_again = guard.evaluate(weight_store, proposals)
    deterministic = result_again.to_dict() == result.to_dict()

    # Lagoon-consistency: conflicting candidates are preserved (in review), not
    # merged into one — the review queue keeps both divergent candidates distinct.
    divergent_in_review = [c for c in result.review_queue if c.path_id == pid_div]
    preserve_both_distinct = len(divergent_in_review) == 2 and len({c.new_value for c in divergent_in_review}) == 2

    guard.write_artifacts(result, output_dir)

    summary = {
        "stage": "stage156_helix_weight_conflict_guard",
        "version": "v0.00.07",
        "candidate_count": len(proposals),
        "clean_count": len(result.clean_candidates),
        "review_count": len(result.review_queue),
        "conflict_count": len(result.conflict_records),
        "stage_gates": {
            "read_only": read_only,
            "no_candidate_lost": no_candidate_lost,
            "divergent_preserve_both": divergent_preserve_both,
            "divergent_not_auto_applied": divergent_not_in_clean,
            "oscillating_detected": oscillating_detected,
            "oscillating_not_auto_applied": oscillating_not_in_clean,
            "clean_passes_through": clean_passes_through,
            "preserve_both_distinct_in_review": preserve_both_distinct,
            "deterministic": deterministic,
            "lagoon_semantic_consistent": preserve_both_distinct and divergent_not_in_clean,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    guard.write_artifacts(result, output_dir, summary=summary)
    return summary
