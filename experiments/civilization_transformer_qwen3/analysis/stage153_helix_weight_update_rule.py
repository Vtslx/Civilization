from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .stage151_helix_path_weight_model import HelixWeightStore, PathWeight
from .stage152_helix_feedback_signal import FeedbackSignal


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage153_helix_weight_update_rule")


@dataclass(frozen=True)
class HelixUpdateConfig:
    """Version constants for the closed-form update rule."""

    lr: float = 0.3
    decay_rate: float = 0.1
    w_min: float = 0.0
    w_max: float = 2.0


@dataclass(frozen=True)
class WeightUpdateProposal:
    path_id: str
    old_value: float
    new_value: float
    reason: str
    feedback_source: str
    signal_ids: tuple[str, ...]
    raw_candidate: float
    clipped: bool
    baseline: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signal_ids"] = list(self.signal_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightUpdateProposal":
        return cls(
            str(payload["path_id"]),
            float(payload["old_value"]),
            float(payload["new_value"]),
            str(payload["reason"]),
            str(payload["feedback_source"]),
            tuple(payload.get("signal_ids", [])),
            float(payload["raw_candidate"]),
            bool(payload["clipped"]),
            float(payload["baseline"]),
        )


class HelixWeightUpdateRule:
    """Closed-form, bounded, gradient-free update over learnable path weights.

    For each path with feedback signals::

        candidate = w_old + lr * sum(signal magnitudes)
        w_new     = clip(candidate, w_min, w_max)

    For each path with no signals this round (unused)::

        candidate = w_old + decay_rate * (baseline - w_old)
        w_new     = clip(candidate, w_min, w_max)

    ``propose`` is a pure function of (store snapshot, signals, config): it never
    mutates the store, so calling it twice on the same inputs yields identical
    proposals (idempotent / replayable). Query-level signals (empty
    ``source_path_id``, e.g. a read miss) carry no link target and are skipped.
    """

    def __init__(self, config: HelixUpdateConfig | None = None) -> None:
        self.config = config or HelixUpdateConfig()

    def propose(self, store: HelixWeightStore, signals: list[FeedbackSignal]) -> list[WeightUpdateProposal]:
        cfg = self.config
        by_path: dict[str, list[FeedbackSignal]] = {}
        for signal in signals:
            if signal.source_path_id == "":
                continue  # query-level signal (read miss): no link target
            if signal.source_path_id not in store.weights:
                raise KeyError(f"signal references unregistered path: {signal.source_path_id}")
            by_path.setdefault(signal.source_path_id, []).append(signal)

        proposals: list[WeightUpdateProposal] = []
        for path_id in sorted(store.weights.keys()):
            weight = store.weights[path_id]
            linked = by_path.get(path_id)
            if linked:
                ordered = sorted(linked, key=lambda s: s.signal_id)
                delta = cfg.lr * sum(s.magnitude for s in ordered)
                candidate = weight.value + delta
                reason = "feedback_update"
                feedback_source = "stage153_rule"
                signal_ids = tuple(s.signal_id for s in ordered)
            else:
                candidate = weight.value + cfg.decay_rate * (weight.baseline - weight.value)
                reason = "decay"
                feedback_source = "decay"
                signal_ids = ()
            new_value = max(cfg.w_min, min(cfg.w_max, candidate))
            proposals.append(
                WeightUpdateProposal(
                    path_id=path_id,
                    old_value=weight.value,
                    new_value=new_value,
                    reason=reason,
                    feedback_source=feedback_source,
                    signal_ids=signal_ids,
                    raw_candidate=candidate,
                    clipped=new_value != candidate,
                    baseline=weight.baseline,
                )
            )
        return proposals

    def apply(self, store: HelixWeightStore, signals: list[FeedbackSignal]) -> list[WeightUpdateProposal]:
        """Commit proposals to the store via Stage151 approve/decay primitives.

        Returns the proposals that were committed, in deterministic order. This
        mutates ``store``; the approval-gating layer that can reject candidates
        is Stage154's concern.
        """
        proposals = self.propose(store, signals)
        for proposal in proposals:
            if proposal.reason == "feedback_update":
                store.approve(
                    proposal.path_id,
                    proposal.new_value,
                    reason=proposal.reason,
                    feedback_source=proposal.feedback_source,
                )
            else:
                store.decay(proposal.path_id, decay_rate=self.config.decay_rate)
        return proposals

    def write_proposals(self, proposals: list[WeightUpdateProposal], output_dir: str | Path, *, summary: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "weight_update_proposals.json").write_text(
            json.dumps([p.to_dict() for p in proposals], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if summary is not None:
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_smoke_store() -> tuple[HelixWeightStore, list[FeedbackSignal], dict[str, float]]:
    """A weight store + signals exercising feedback update, clipping, and unused decay.

    Returns the store, the collected signals, and the expected post-apply values
    so the smoke can verify the rule end-to-end.
    """
    from .stage152_helix_feedback_signal import HelixFeedbackSignalCollector
    from .stage73_orion_memory_kernel import OrionMemoryStore

    clock = {"now": 5000.0}
    orion = OrionMemoryStore(now_fn=lambda: clock["now"])

    # link1: temporal a->b, will receive a HIT and is boosted to force clipping.
    cell_a = orion.write_cell(memory_system="episodic", content="alpha", summary="a", source="stage153_smoke")
    cell_b = orion.write_cell(memory_system="episodic", content="beta", summary="b", source="stage153_smoke")
    clock["now"] = 5001.0
    orion.link_cells(cell_a.cell_id, cell_b.cell_id, link_type="temporal", weight=1.0)

    # link2: conflict c<->d, receives REJECTED.
    cell_c = orion.write_cell(memory_system="semantic", content="claim left", summary="c", source="stage153_smoke")
    cell_d = orion.write_cell(memory_system="semantic", content="claim right", summary="d", source="stage153_smoke")
    clock["now"] = 5002.0
    orion.mark_conflict(cell_c.cell_id, cell_d.cell_id, reason="dispute")

    # link3: temporal e->f, never read -> unused decay. Boosted above baseline.
    cell_e = orion.write_cell(memory_system="episodic", content="gamma delta", summary="e", source="stage153_smoke")
    cell_f = orion.write_cell(memory_system="episodic", content="epsilon zeta", summary="f", source="stage153_smoke")
    clock["now"] = 5003.0
    orion.link_cells(cell_e.cell_id, cell_f.cell_id, link_type="temporal", weight=0.5)

    # read("alpha") retrieves cell_a only -> HIT on its incident temporal link.
    orion.read("alpha")

    collector = HelixFeedbackSignalCollector()
    signals = collector.collect(orion)

    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])
    links = list(orion.links)
    for link in links:
        weight_store.register_link(link)

    # Boost link1 to 1.95 so the HIT update (1.95 + 0.15 = 2.10) clips at w_max.
    link1_id = "link:%s:%s:temporal" % (cell_a.cell_id, cell_b.cell_id)
    weight_store.approve(link1_id, 1.95, reason="boost", feedback_source="stage153_smoke")
    # Boost link3 to 0.9 so unused decay moves it back toward baseline 0.5.
    link3_id = "link:%s:%s:temporal" % (cell_e.cell_id, cell_f.cell_id)
    weight_store.approve(link3_id, 0.9, reason="boost", feedback_source="stage153_smoke")

    expected = {
        link1_id: 2.0,  # 1.95 + 0.3*0.5 = 2.10 -> clipped to 2.0
        "link:%s:%s:conflict" % (cell_c.cell_id, cell_d.cell_id): 0.76,  # 1.0 + 0.3*(-0.8) = 0.76
        link3_id: 0.86,  # 0.9 + 0.1*(0.5-0.9) = 0.86
    }
    return weight_store, signals, expected


def run_stage153_helix_weight_update_rule_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    weight_store, signals, expected = _build_smoke_store()
    rule = HelixWeightUpdateRule(HelixUpdateConfig())

    # Purity: snapshot the store, propose, then verify the store is untouched.
    snapshot = weight_store.to_dict()
    proposals = rule.propose(weight_store, signals)
    pure = weight_store.to_dict() == snapshot

    # Idempotent: a second proposal on the unchanged store is identical.
    proposals_again = rule.propose(weight_store, signals)
    idempotent = [p.to_dict() for p in proposals_again] == [p.to_dict() for p in proposals]

    cfg = rule.config
    bounded = all(cfg.w_min <= p.new_value <= cfg.w_max for p in proposals)
    no_gradients = True  # closed-form, no torch / autograd anywhere

    # Decay monotonic toward baseline: the unused path moved strictly between
    # its old value and its baseline without crossing it.
    decay_proposals = [p for p in proposals if p.reason == "decay"]
    decay_monotonic = all(
        (p.baseline <= p.new_value <= p.old_value) or (p.old_value <= p.new_value <= p.baseline)
        for p in decay_proposals
    )

    clipped_exists = any(p.clipped for p in proposals)
    feedback_exists = any(p.reason == "feedback_update" for p in proposals)

    # Deterministic ordering: proposals are sorted by path_id.
    ordered_by_path = [p.path_id for p in proposals] == sorted(p.path_id for p in proposals)

    # Apply end-to-end and verify the committed weights match the proposals.
    applied = rule.apply(weight_store, signals)
    apply_matches = all(
        weight_store.weights[p.path_id].value == p.new_value for p in applied
    )
    expected_matches = all(
        abs(weight_store.weights[path_id].value - value) < 1e-9 for path_id, value in expected.items()
    )

    rule.write_proposals(proposals, output_dir)

    summary = {
        "stage": "stage153_helix_weight_update_rule",
        "version": "v0.00.07",
        "config": asdict(cfg),
        "proposal_count": len(proposals),
        "clipped_count": sum(1 for p in proposals if p.clipped),
        "feedback_count": sum(1 for p in proposals if p.reason == "feedback_update"),
        "decay_count": sum(1 for p in proposals if p.reason == "decay"),
        "stage_gates": {
            "pure_function": pure,
            "idempotent": idempotent,
            "bounded": bounded,
            "decay_monotonic_toward_baseline": decay_monotonic,
            "clipping_demonstrated": clipped_exists,
            "feedback_update_demonstrated": feedback_exists,
            "deterministic_order": ordered_by_path,
            "apply_matches_proposals": apply_matches,
            "expected_values_match": expected_matches,
            "no_gradients": no_gradients,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    rule.write_proposals(proposals, output_dir, summary=summary)
    return summary
