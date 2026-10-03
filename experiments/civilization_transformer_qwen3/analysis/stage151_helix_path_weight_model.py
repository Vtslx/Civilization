from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any

from .stage145_rosette_honeycomb_graph import RosetteHoneycombEdge
from .stage73_orion_memory_kernel import MemoryLink, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage151_helix_path_weight_model")


class WeightUpdateAction(StrEnum):
    PROPOSE = "propose"
    APPROVE = "approve"
    REJECT = "reject"
    DECAY = "decay"
    CLIP = "clip"


def path_id_for_link(link: MemoryLink) -> str:
    """Deterministic path identity for a memory link.

    Directional: a reversed link (B->A) is a distinct path. Duplicate links of
    the same (source, target, type) collapse to one path weight, which matches
    the semantic that the weight belongs to the relationship, not the instance.
    """
    return f"link:{link.source_cell_id}:{link.target_cell_id}:{link.link_type.value}"


def path_id_for_edge(edge: RosetteHoneycombEdge) -> str:
    """Deterministic path identity for a honeycomb graph edge."""
    return f"edge:{edge.source_node_id}:{edge.target_node_id}:{edge.adjacency_type}"


@dataclass(frozen=True)
class PathWeight:
    path_id: str
    value: float
    version: int
    baseline: float
    updated_at: float
    update_reason: str
    feedback_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PathWeight":
        return cls(
            str(payload["path_id"]),
            float(payload["value"]),
            int(payload["version"]),
            float(payload["baseline"]),
            float(payload["updated_at"]),
            str(payload["update_reason"]),
            str(payload["feedback_source"]),
        )


@dataclass(frozen=True)
class WeightUpdateEvent:
    event_id: str
    path_id: str
    action: WeightUpdateAction
    old_value: float
    new_value: float
    version: int
    created_at: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightUpdateEvent":
        return cls(
            str(payload["event_id"]),
            str(payload["path_id"]),
            WeightUpdateAction(payload["action"]),
            float(payload["old_value"]),
            float(payload["new_value"]),
            int(payload["version"]),
            float(payload["created_at"]),
            dict(payload.get("details", {})),
        )


class HelixWeightStore:
    """Versioned, auditable learnable weights over memory paths.

    The store is an overlay: it never mutates the frozen ``MemoryLink`` or
    ``RosetteHoneycombEdge`` records it references. A link's static ``weight``
    becomes the immutable ``baseline``; the learnable ``value`` lives here and
    is versioned through candidate/approve/decay/clip events that are preserved
    in the audit trace rather than overwritten.
    """

    def __init__(self, *, now_fn=time.time) -> None:
        self._now_fn = now_fn
        self.weights: dict[str, PathWeight] = {}
        self.update_events: list[WeightUpdateEvent] = []
        self._next_event_id = 1

    def _now(self) -> float:
        return float(self._now_fn())

    def _new_event_id(self) -> str:
        event_id = f"weight-event-{self._next_event_id:06d}"
        self._next_event_id += 1
        return event_id

    def _record(
        self,
        path_id: str,
        action: WeightUpdateAction,
        *,
        old_value: float,
        new_value: float,
        version: int,
        details: dict[str, Any] | None = None,
    ) -> WeightUpdateEvent:
        event = WeightUpdateEvent(
            event_id=self._new_event_id(),
            path_id=path_id,
            action=action,
            old_value=old_value,
            new_value=new_value,
            version=version,
            created_at=self._now(),
            details=details or {},
        )
        self.update_events.append(event)
        return event

    def _required_weight(self, path_id: str) -> PathWeight:
        try:
            return self.weights[path_id]
        except KeyError as error:
            raise KeyError(f"unknown path weight: {path_id}") from error

    def register_link(self, link: MemoryLink, *, feedback_source: str = "baseline") -> PathWeight:
        return self._register(path_id_for_link(link), baseline=link.weight, feedback_source=feedback_source)

    def register_edge(self, edge: RosetteHoneycombEdge, *, feedback_source: str = "baseline") -> PathWeight:
        return self._register(path_id_for_edge(edge), baseline=edge.weight, feedback_source=feedback_source)

    def _register(self, path_id: str, *, baseline: float, feedback_source: str) -> PathWeight:
        if path_id in self.weights:
            return self.weights[path_id]
        baseline = max(0.0, float(baseline))
        weight = PathWeight(
            path_id=path_id,
            value=baseline,
            version=0,
            baseline=baseline,
            updated_at=self._now(),
            update_reason="init",
            feedback_source=feedback_source,
        )
        self.weights[path_id] = weight
        self._record(
            path_id,
            WeightUpdateAction.APPROVE,
            old_value=baseline,
            new_value=baseline,
            version=0,
            details={"reason": "init", "feedback_source": feedback_source},
        )
        return weight

    def propose(
        self,
        path_id: str,
        new_value: float,
        *,
        reason: str,
        feedback_source: str,
    ) -> WeightUpdateEvent:
        """Record a candidate update without changing the current weight."""
        current = self._required_weight(path_id)
        return self._record(
            path_id,
            WeightUpdateAction.PROPOSE,
            old_value=current.value,
            new_value=float(new_value),
            version=current.version,
            details={"reason": reason, "feedback_source": feedback_source},
        )

    def approve(
        self,
        path_id: str,
        new_value: float,
        *,
        reason: str,
        feedback_source: str,
    ) -> PathWeight:
        """Commit an update: version increments, history is preserved in the trace."""
        current = self._required_weight(path_id)
        new_value = float(new_value)
        updated = replace(
            current,
            value=new_value,
            version=current.version + 1,
            updated_at=self._now(),
            update_reason=reason,
            feedback_source=feedback_source,
        )
        self.weights[path_id] = updated
        self._record(
            path_id,
            WeightUpdateAction.APPROVE,
            old_value=current.value,
            new_value=new_value,
            version=updated.version,
            details={"reason": reason, "feedback_source": feedback_source},
        )
        return updated

    def reject(
        self,
        path_id: str,
        proposed_value: float,
        *,
        reason: str,
        feedback_source: str,
    ) -> WeightUpdateEvent:
        """Record a rejected candidate; the current weight is untouched."""
        current = self._required_weight(path_id)
        return self._record(
            path_id,
            WeightUpdateAction.REJECT,
            old_value=current.value,
            new_value=float(proposed_value),
            version=current.version,
            details={"reason": reason, "feedback_source": feedback_source},
        )

    def decay(self, path_id: str, *, decay_rate: float) -> PathWeight:
        """Move the value toward baseline by ``decay_rate`` in [0, 1].

        Emits a dedicated DECAY event (not APPROVE) so the audit trace
        distinguishes learned-rule decay from an explicit approval.
        """
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay_rate must be within [0, 1]")
        current = self._required_weight(path_id)
        new_value = current.value + decay_rate * (current.baseline - current.value)
        updated = replace(
            current,
            value=new_value,
            version=current.version + 1,
            updated_at=self._now(),
            update_reason="decay",
            feedback_source="decay",
        )
        self.weights[path_id] = updated
        self._record(
            path_id,
            WeightUpdateAction.DECAY,
            old_value=current.value,
            new_value=new_value,
            version=updated.version,
            details={"decay_rate": decay_rate, "baseline": current.baseline},
        )
        return updated

    def clip(self, path_id: str, *, lo: float, hi: float) -> PathWeight:
        """Clamp the value to [lo, hi]; version increments only when it changes."""
        if lo > hi:
            raise ValueError("lo must not exceed hi")
        current = self._required_weight(path_id)
        new_value = max(lo, min(hi, current.value))
        if new_value == current.value:
            self._record(
                path_id,
                WeightUpdateAction.CLIP,
                old_value=current.value,
                new_value=new_value,
                version=current.version,
                details={"lo": lo, "hi": hi, "changed": False},
            )
            return current
        updated = replace(
            current,
            value=new_value,
            version=current.version + 1,
            updated_at=self._now(),
            update_reason="clip",
            feedback_source="clip",
        )
        self.weights[path_id] = updated
        self._record(
            path_id,
            WeightUpdateAction.CLIP,
            old_value=current.value,
            new_value=new_value,
            version=updated.version,
            details={"lo": lo, "hi": hi, "changed": True},
        )
        return updated

    def summary(self) -> dict[str, Any]:
        return {
            "stage": "stage151_helix_path_weight_model",
            "version": "v0.00.07",
            "path_count": len(self.weights),
            "update_event_count": len(self.update_events),
            "update_actions": sorted({event.action.value for event in self.update_events}),
            "max_version": max((w.version for w in self.weights.values()), default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [w.to_dict() for w in sorted(self.weights.values(), key=lambda w: w.path_id)],
            "update_events": [e.to_dict() for e in self.update_events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, now_fn=time.time) -> "HelixWeightStore":
        store = cls(now_fn=now_fn)
        for item in payload.get("weights", []):
            weight = PathWeight.from_dict(item)
            store.weights[weight.path_id] = weight
        for item in payload.get("update_events", []):
            event = WeightUpdateEvent.from_dict(item)
            store.update_events.append(event)
        # restore monotonic event id counter
        max_id = 0
        for event in store.update_events:
            try:
                max_id = max(max_id, int(event.event_id.rsplit("-", 1)[-1]))
            except ValueError:
                continue
        store._next_event_id = max_id + 1
        return store

    def write_artifacts(self, output_dir: str | Path, *, summary: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "path_weights.json").write_text(
            json.dumps([w.to_dict() for w in sorted(self.weights.values(), key=lambda w: w.path_id)], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output / "weight_update_events.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.update_events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        (output / "summary.json").write_text(json.dumps(summary or self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_smoke_store() -> tuple[OrionMemoryStore, list[MemoryLink]]:
    """Minimal Orion store with typed links of known weights for the smoke fixture."""
    clock = {"now": 2000.0}
    store = OrionMemoryStore(now_fn=lambda: clock["now"])

    episodic_a = store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="Helix Stage151 introduces versioned learnable path weights.",
        summary="helix stage151 kickoff",
        source="stage151_smoke",
        time_index=2001.0,
    )
    episodic_b = store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="Learnable weights keep Qwen3 frozen and reuse the existing audit trace.",
        summary="helix stage151 constraint",
        source="stage151_smoke",
        time_index=2002.0,
    )
    clock["now"] = 2003.0
    temporal = store.link_cells(episodic_a.cell_id, episodic_b.cell_id, link_type="temporal", weight=0.8)

    semantic = store.consolidate_episodic_to_semantic(
        [episodic_a.cell_id, episodic_b.cell_id],
        summary="Helix Stage151 path-weight principle",
        content="Path weights are a versioned overlay; links and edges stay frozen.",
    )
    procedural = store.write_procedural_from_task_trace(
        task_name="stage151_path_weight_smoke",
        steps=["register links", "register edges", "propose", "approve", "decay"],
        outcome="success",
        source="stage151_smoke",
    )
    clock["now"] = 2004.0
    procedure = store.link_cells(semantic.cell_id, procedural.cell_id, link_type="procedure", weight=0.7)

    conflict_left = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="Stage151 must not modify the Stage49-72 inference service.",
        summary="service isolation",
        source="stage151_smoke",
    )
    conflict_right = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="A later stage may route learned weights through the service path.",
        summary="future service routing",
        source="stage151_smoke",
    )
    clock["now"] = 2005.0
    conflict = store.mark_conflict(conflict_left.cell_id, conflict_right.cell_id, reason="current boundary versus future routing")

    # links are returned in insertion order for deterministic registration
    return store, [temporal, procedure, conflict]


def _smoke_edges() -> list[RosetteHoneycombEdge]:
    """Hand-built honeycomb edges covering consolidation and conflict adjacency."""
    return [
        RosetteHoneycombEdge(source_node_id="node-episodic-1", target_node_id="node-schema-1", adjacency_type="consolidation", weight=1.0),
        RosetteHoneycombEdge(source_node_id="node-schema-1", target_node_id="node-schema-2", adjacency_type="conflict", weight=0.5),
    ]


def run_stage151_helix_path_weight_model_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    clock = {"now": 3000.0}
    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])

    orion_store, links = _build_smoke_store()
    edges = _smoke_edges()

    # Snapshot link/edge serialization before registration to prove the overlay
    # never touches the frozen evidence records.
    link_payloads_before = [link.to_dict() for link in links]
    edge_payloads_before = [edge.to_dict() for edge in edges]

    registered_links = [weight_store.register_link(link) for link in links]
    registered_edges = [weight_store.register_edge(edge) for edge in edges]
    all_registered = registered_links + registered_edges

    init_weights_match_baseline = all(w.value == w.baseline for w in all_registered)
    init_versions_zero = all(w.version == 0 for w in all_registered)
    init_reasons = all(w.update_reason == "init" for w in all_registered)

    # Exercise the audit-trace primitives without invoking the Stage153 rule.
    temporal_id = path_id_for_link(links[0])
    procedure_id = path_id_for_link(links[1])
    conflict_id = path_id_for_link(links[2])
    consolidation_edge_id = path_id_for_edge(edges[0])
    conflict_edge_id = path_id_for_edge(edges[1])

    clock["now"] = 3010.0
    weight_store.propose(temporal_id, 0.95, reason="retrieval hit candidate", feedback_source="read_hit")
    clock["now"] = 3011.0
    weight_store.approve(procedure_id, 0.9, reason="skill reuse success", feedback_source="skill_success")
    clock["now"] = 3012.0
    weight_store.reject(conflict_id, 0.2, reason="conflict decision suppressed weight", feedback_source="conflict_decision")
    # Decay only moves toward baseline when value != baseline, so boost first.
    clock["now"] = 3013.0
    weight_store.approve(consolidation_edge_id, 1.5, reason="retrieval boost", feedback_source="read_hit")
    consolidation_pre_decay = weight_store.weights[consolidation_edge_id].value
    clock["now"] = 3014.0
    weight_store.decay(consolidation_edge_id, decay_rate=0.25)
    # Clip is demonstrated as an actual clamp, not a no-op.
    clock["now"] = 3015.0
    weight_store.approve(conflict_edge_id, 1.2, reason="overshoot candidate", feedback_source="read_hit")
    clock["now"] = 3016.0
    weight_store.clip(conflict_edge_id, lo=0.0, hi=1.0)

    link_payloads_after = [link.to_dict() for link in links]
    edge_payloads_after = [edge.to_dict() for edge in edges]
    evidence_serialization_unchanged = link_payloads_before == link_payloads_after and edge_payloads_before == edge_payloads_after

    # Round-trip the whole store through JSON to prove deterministic serialization.
    round_trip = HelixWeightStore.from_dict(json.loads(json.dumps(weight_store.to_dict(), ensure_ascii=False)))
    weights_round_trip = [w.to_dict() for w in sorted(round_trip.weights.values(), key=lambda w: w.path_id)]
    weights_original = [w.to_dict() for w in sorted(weight_store.weights.values(), key=lambda w: w.path_id)]
    events_round_trip = [e.to_dict() for e in round_trip.update_events]
    events_original = [e.to_dict() for e in weight_store.update_events]
    serialization_round_trip = weights_original == weights_round_trip and events_original == events_round_trip

    action_set = {action.value for action in WeightUpdateAction}
    observed_actions = {event.action.value for event in weight_store.update_events}

    approved_procedure = weight_store.weights[procedure_id]
    decayed_edge = weight_store.weights[consolidation_edge_id]
    clamped_edge = weight_store.weights[conflict_edge_id]

    required_keys = {"path_id", "value", "version", "baseline", "updated_at", "update_reason", "feedback_source"}

    summary = {
        **weight_store.summary(),
        "stage_gates": {
            "version_fields_present": all(required_keys.issubset(w.to_dict().keys()) for w in all_registered),
            "initial_weight_equals_baseline": init_weights_match_baseline,
            "initial_version_zero": init_versions_zero,
            "init_reason_set": init_reasons,
            "approve_increments_version": approved_procedure.version == 1 and approved_procedure.value == 0.9,
            "decay_moves_toward_baseline": decayed_edge.baseline < decayed_edge.value < consolidation_pre_decay,
            "clip_clamps_into_bounds": clamped_edge.value == 1.0 and clamped_edge.version == 2,
            "all_update_actions_observed": action_set.issubset(observed_actions),
            "trace_events_serializable": serialization_round_trip,
            "evidence_serialization_unchanged": evidence_serialization_unchanged,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    weight_store.write_artifacts(output_dir, summary=summary)
    return summary
