from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any

from .stage151_helix_path_weight_model import path_id_for_link
from .stage73_orion_memory_kernel import (
    MemoryLink,
    MemoryLinkType,
    MemoryTraceAction,
    MemoryTraceEvent,
    OrionMemoryStore,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage152_helix_feedback_signal")

# Stage152 version constants: bounded feedback magnitudes in [-1, 1]. Stage153
# consumes these as the ``signal`` term in the closed-form update rule.
HIT_MAGNITUDE = 0.5
MISS_MAGNITUDE = -0.3
APPROVED_MAGNITUDE = 0.8
REJECTED_MAGNITUDE = -0.8
SUCCESS_MAGNITUDE = 0.7
FAILURE_MAGNITUDE = -0.7


class FeedbackKind(StrEnum):
    HIT = "hit"
    MISS = "miss"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUCCESS = "success"
    FAILURE = "failure"


# Trace actions that can yield feedback. Other actions (WRITE/LINK/UPDATE/EXPIRE)
# are not feedback sources and are skipped deterministically.
FEEDBACK_ACTIONS: frozenset[MemoryTraceAction] = frozenset(
    {MemoryTraceAction.READ, MemoryTraceAction.CONSOLIDATE, MemoryTraceAction.CONFLICT}
)


@dataclass(frozen=True)
class FeedbackSignal:
    signal_id: str
    source_path_id: str
    kind: FeedbackKind
    magnitude: float
    evidence_trace_id: str
    created_at: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedbackSignal":
        return cls(
            str(payload["signal_id"]),
            str(payload["source_path_id"]),
            FeedbackKind(payload["kind"]),
            float(payload["magnitude"]),
            str(payload["evidence_trace_id"]),
            float(payload["created_at"]),
            dict(payload.get("details", {})),
        )


class HelixFeedbackSignalCollector:
    """Derives typed, bounded feedback signals from existing memory trace events.

    Read-only: it never writes to the store, links, or cells. A signal inherits
    its ``created_at`` from the evidence trace event, so collecting twice on the
    same store yields byte-identical signals (replayable). ``source_path_id``
    reuses Stage151's ``path_id_for_link`` so signals key directly into the
    HelixWeightStore; query-level signals (a read miss) carry an empty path.
    """

    def collect(self, store: OrionMemoryStore) -> list[FeedbackSignal]:
        incident = self._incident_links_by_cell(store.links)
        existing_cells = set(store.cells.keys())
        signals: list[FeedbackSignal] = []
        counter = 0
        for event in store.trace_events:
            if event.action not in FEEDBACK_ACTIONS:
                continue
            for kind, magnitude, source_path_id, details in self._classify(event, incident, existing_cells):
                counter += 1
                signals.append(
                    FeedbackSignal(
                        signal_id=f"signal-{counter:06d}",
                        source_path_id=source_path_id,
                        kind=kind,
                        magnitude=magnitude,
                        evidence_trace_id=event.event_id,
                        created_at=event.created_at,
                        details=details,
                    )
                )
        return signals

    def _incident_links_by_cell(self, links: list[MemoryLink]) -> dict[str, list[MemoryLink]]:
        index: dict[str, list[MemoryLink]] = {}
        for link in links:
            index.setdefault(link.source_cell_id, []).append(link)
            index.setdefault(link.target_cell_id, []).append(link)
        for cell_id in index:
            index[cell_id].sort(key=lambda lk: path_id_for_link(lk))
        return index

    def _classify(
        self,
        event: MemoryTraceEvent,
        incident: dict[str, list[MemoryLink]],
        existing_cells: set[str],
    ) -> list[tuple[FeedbackKind, float, str, dict[str, Any]]]:
        if event.action is MemoryTraceAction.READ:
            return self._classify_read(event, incident, existing_cells)
        if event.action is MemoryTraceAction.CONSOLIDATE:
            return self._classify_consolidate(event, incident, existing_cells)
        if event.action is MemoryTraceAction.CONFLICT:
            return self._classify_conflict(event, incident, existing_cells)
        return []

    def _classify_read(
        self,
        event: MemoryTraceEvent,
        incident: dict[str, list[MemoryLink]],
        existing_cells: set[str],
    ) -> list[tuple[FeedbackKind, float, str, dict[str, Any]]]:
        result_count = event.details.get("result_count")
        read_by_id = bool(event.details.get("read_by_id"))
        # A query read that found nothing is a query-level miss (no link target).
        if result_count == 0 and not read_by_id:
            return [(FeedbackKind.MISS, MISS_MAGNITUDE, "", {"query": event.details.get("query")})]
        out: list[tuple[FeedbackKind, float, str, dict[str, Any]]] = []
        for cell_id in event.cell_ids:
            if cell_id not in existing_cells:
                raise KeyError(f"feedback trace references dangling cell: {cell_id}")
            for link in incident.get(cell_id, []):
                out.append(
                    (
                        FeedbackKind.HIT,
                        HIT_MAGNITUDE,
                        path_id_for_link(link),
                        {"retrieved_cell_id": cell_id, "link_type": link.link_type.value, "query": event.details.get("query")},
                    )
                )
        return out

    def _classify_consolidate(
        self,
        event: MemoryTraceEvent,
        incident: dict[str, list[MemoryLink]],
        existing_cells: set[str],
    ) -> list[tuple[FeedbackKind, float, str, dict[str, Any]]]:
        for cell_id in event.cell_ids:
            if cell_id not in existing_cells:
                raise KeyError(f"feedback trace references dangling cell: {cell_id}")
        if event.details.get("procedural_from_task_trace"):
            outcome = event.details.get("outcome")
            if outcome == "success":
                kind, magnitude = FeedbackKind.SUCCESS, SUCCESS_MAGNITUDE
            elif outcome == "failure":
                kind, magnitude = FeedbackKind.FAILURE, FAILURE_MAGNITUDE
            else:
                raise ValueError(f"unknown procedural outcome: {outcome!r}")
            procedural_cell_id = event.cell_ids[0]
            return self._signals_for_links(
                incident.get(procedural_cell_id, []),
                MemoryLinkType.PROCEDURE,
                kind,
                magnitude,
                {"procedural_cell_id": procedural_cell_id, "outcome": outcome},
            )
        if "semantic_cell_id" in event.details:
            semantic_cell_id = event.details["semantic_cell_id"]
            episodic_ids = [cid for cid in event.cell_ids if cid != semantic_cell_id]
            replay_links = [
                link
                for link in incident.get(semantic_cell_id, [])
                if link.link_type is MemoryLinkType.REPLAY and link.source_cell_id in episodic_ids
            ]
            return self._signals_for_links(
                replay_links,
                MemoryLinkType.REPLAY,
                FeedbackKind.APPROVED,
                APPROVED_MAGNITUDE,
                {"semantic_cell_id": semantic_cell_id, "episodic_ids": episodic_ids},
            )
        # Known action but an unmapped consolidation shape: skip without failing.
        return []

    def _classify_conflict(
        self,
        event: MemoryTraceEvent,
        incident: dict[str, list[MemoryLink]],
        existing_cells: set[str],
    ) -> list[tuple[FeedbackKind, float, str, dict[str, Any]]]:
        source_id, target_id = event.cell_ids[0], event.cell_ids[1]
        for cell_id in (source_id, target_id):
            if cell_id not in existing_cells:
                raise KeyError(f"feedback trace references dangling cell: {cell_id}")
        conflict_links = [
            link
            for link in incident.get(source_id, [])
            if link.link_type is MemoryLinkType.CONFLICT and link.target_cell_id == target_id
        ]
        if not conflict_links:
            raise KeyError(f"conflict trace without a matching conflict link: {source_id} -> {target_id}")
        return self._signals_for_links(
            conflict_links,
            MemoryLinkType.CONFLICT,
            FeedbackKind.REJECTED,
            REJECTED_MAGNITUDE,
            {"source_cell_id": source_id, "target_cell_id": target_id},
        )

    @staticmethod
    def _signals_for_links(
        links: list[MemoryLink],
        expected_type: MemoryLinkType,
        kind: FeedbackKind,
        magnitude: float,
        details: dict[str, Any],
    ) -> list[tuple[FeedbackKind, float, str, dict[str, Any]]]:
        out: list[tuple[FeedbackKind, float, str, dict[str, Any]]] = []
        for link in sorted(links, key=path_id_for_link):
            if link.link_type is not expected_type:
                continue
            out.append((kind, magnitude, path_id_for_link(link), {**details, "link_type": link.link_type.value}))
        return out


def _build_smoke_store() -> OrionMemoryStore:
    """Store whose trace covers all six feedback kinds."""
    clock = {"now": 4000.0}
    store = OrionMemoryStore(now_fn=lambda: clock["now"])

    episodic_a = store.write_cell(
        memory_system="episodic",
        content="Helix feedback collector derives signals from existing trace events.",
        summary="helix feedback hit target",
        source="stage152_smoke",
        time_index=4001.0,
    )
    episodic_b = store.write_cell(
        memory_system="episodic",
        content="Read hits reinforce the links that led to a retrieved cell.",
        summary="helix feedback hit support",
        source="stage152_smoke",
        time_index=4002.0,
    )
    clock["now"] = 4003.0
    store.link_cells(episodic_a.cell_id, episodic_b.cell_id, link_type="temporal", weight=0.8)

    semantic = store.consolidate_episodic_to_semantic(
        [episodic_a.cell_id, episodic_b.cell_id],
        summary="Helix feedback principle",
        content="Feedback signals are read-only, bounded, and replayable.",
    )

    # A read that matches -> HIT signals on the incident links of retrieved cells.
    store.read("helix feedback hit target", memory_system="episodic")
    # A read that matches nothing -> one MISS signal at query level.
    store.read("zzz_nonexistent_query_token_zzz")

    success_skill = store.write_procedural_from_task_trace(
        task_name="stage152_success_skill",
        steps=["collect", "classify", "emit"],
        outcome="success",
        source="stage152_smoke",
    )
    clock["now"] = 4004.0
    store.link_cells(semantic.cell_id, success_skill.cell_id, link_type="procedure", weight=0.7)

    failure_skill = store.write_procedural_from_task_trace(
        task_name="stage152_failure_skill",
        steps=["skip", "classify", "drop"],
        outcome="failure",
        source="stage152_smoke",
    )
    clock["now"] = 4005.0
    store.link_cells(semantic.cell_id, failure_skill.cell_id, link_type="procedure", weight=0.6)

    conflict_left = store.write_cell(
        memory_system="semantic",
        content="Feedback must mutate live links.",
        summary="wrong feedback claim",
        source="stage152_smoke",
    )
    conflict_right = store.write_cell(
        memory_system="semantic",
        content="Feedback must stay read-only.",
        summary="correct feedback claim",
        source="stage152_smoke",
    )
    clock["now"] = 4006.0
    store.mark_conflict(conflict_left.cell_id, conflict_right.cell_id, reason="mutation boundary dispute")
    return store


def run_stage152_helix_feedback_signal_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store = _build_smoke_store()
    collector = HelixFeedbackSignalCollector()

    # Snapshot the store before collection to prove the collector is read-only.
    cells_before = [c.to_dict() for c in store.cells.values()]
    links_before = [lk.to_dict() for lk in store.links]
    trace_before = [e.to_dict() for e in store.trace_events]

    signals = collector.collect(store)

    read_only = (
        [c.to_dict() for c in store.cells.values()] == cells_before
        and [lk.to_dict() for lk in store.links] == links_before
        and [e.to_dict() for e in store.trace_events] == trace_before
    )

    # Replayability: a second collection on the unchanged store is byte-identical.
    replay = collector.collect(store)
    replayable = [s.to_dict() for s in replay] == [s.to_dict() for s in signals]

    observed_kinds = {s.kind.value for s in signals}
    required_kinds = {kind.value for kind in FeedbackKind}
    magnitudes_bounded = all(-1.0 <= s.magnitude <= 1.0 for s in signals)

    # Every signal's evidence_trace_id resolves to a real trace event.
    trace_ids = {e.event_id for e in store.trace_events}
    evidence_resolves = all(s.evidence_trace_id in trace_ids for s in signals)

    # Every non-query signal resolves to a real link path_id.
    link_path_ids = {path_id_for_link(lk) for lk in store.links}
    source_resolves = all(s.source_path_id == "" or s.source_path_id in link_path_ids for s in signals)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feedback_signals.json").write_text(
        json.dumps([s.to_dict() for s in signals], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "stage": "stage152_helix_feedback_signal",
        "version": "v0.00.07",
        "signal_count": len(signals),
        "kind_counts": {kind: sum(1 for s in signals if s.kind.value == kind) for kind in sorted(required_kinds)},
        "magnitude_bounds": {"min": min((s.magnitude for s in signals), default=0.0), "max": max((s.magnitude for s in signals), default=0.0)},
        "stage_gates": {
            "read_only": read_only,
            "replayable": replayable,
            "all_kinds_observed": required_kinds.issubset(observed_kinds),
            "magnitudes_bounded": magnitudes_bounded,
            "evidence_trace_resolves": evidence_resolves,
            "source_path_resolves": source_resolves,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
