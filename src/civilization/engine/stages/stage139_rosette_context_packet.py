from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import EpisodeFrame, TrifidEpisodeBinder
from .stage104_trifid_episode_pattern_completion import TrifidEpisodePatternCompleter
from .stage105_trifid_episode_replay import TrifidEpisodeReplayer
from .stage122_lagoon_consolidation_clusters import LagoonConsolidationClusterer
from .stage123_lagoon_schema_candidates import LagoonSchemaCandidateBuilder
from .stage124_lagoon_approved_consolidation import LagoonApprovedConsolidator
from .stage127_lagoon_schema_conflict_guard import LagoonSchemaConflictGuard
from .stage129_lagoon_conflict_review import LagoonConflictReviewBuilder
from .stage130_lagoon_conflict_decision import LagoonConflictDecisionLedger
from .stage131_lagoon_decision_aware_recall import LagoonDecisionAwareRecallPolicy
from .stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage139_rosette_context_packet")


@dataclass(frozen=True)
class RosetteContextGroup:
    group_id: str
    status: str
    eligible_for_context: bool
    factual_cell_ids: tuple[str, ...]
    control_cell_ids: tuple[str, ...]
    support_episode_ids: tuple[str, ...]
    conflict_episode_ids: tuple[str, ...]
    confidence: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosetteContextGroup":
        return cls(
            group_id=str(payload["group_id"]),
            status=str(payload["status"]),
            eligible_for_context=bool(payload["eligible_for_context"]),
            factual_cell_ids=tuple(payload["factual_cell_ids"]),
            control_cell_ids=tuple(payload["control_cell_ids"]),
            support_episode_ids=tuple(payload["support_episode_ids"]),
            conflict_episode_ids=tuple(payload["conflict_episode_ids"]),
            confidence=float(payload["confidence"]),
            importance=float(payload["importance"]),
        )


@dataclass(frozen=True)
class RosetteContextPacket:
    packet_id: str
    query: str
    groups: tuple[RosetteContextGroup, ...]

    @property
    def factual_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell_id for group in self.groups for cell_id in group.factual_cell_ids)

    @property
    def control_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell_id for group in self.groups for cell_id in group.control_cell_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"packet_id": self.packet_id, "query": self.query, "groups": [group.to_dict() for group in self.groups]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosetteContextPacket":
        groups = tuple(RosetteContextGroup.from_dict(item) for item in payload["groups"])
        return cls(packet_id=str(payload["packet_id"]), query=str(payload["query"]), groups=groups)


class RosetteContextAssembler:
    """Converts verified Lagoon recalls into typed factual/control context groups."""

    def assemble(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, query: str) -> RosetteContextPacket:
        recalls = LagoonDecisionAwareRecallPolicy().retrieve(store, binder, query)
        groups: list[RosetteContextGroup] = []
        for recall in recalls:
            semantic = store.cells.get(recall.semantic_cell_id)
            if semantic is None or semantic.memory_system != MemorySystem.SEMANTIC or semantic.source != "lagoon_stage124":
                raise ValueError("recall semantic cell is not a Lagoon schema")
            factual_cells = store.read_cells(list(recall.context_cell_ids), trace_details={"rosette_context_packet": True, "role": "factual"})
            if len(factual_cells) != len(recall.context_cell_ids) or any(cell.memory_system == MemorySystem.PROCEDURAL for cell in factual_cells):
                raise ValueError("factual context contains missing or procedural cells")
            control_ids = (recall.decision_cell_id,) if recall.decision_cell_id else ()
            if control_ids:
                controls = store.read_cells(list(control_ids), trace_details={"rosette_context_packet": True, "role": "control"})
                if len(controls) != 1 or controls[0].memory_system != MemorySystem.PROCEDURAL:
                    raise ValueError("control reference is not procedural memory")
            support_episode_ids = semantic.metadata.get("lagoon_support_episode_ids")
            if not isinstance(support_episode_ids, list):
                raise ValueError("Lagoon schema is missing support provenance")
            groups.append(RosetteContextGroup(
                group_id=f"rosette:{semantic.cell_id}",
                status=recall.status,
                eligible_for_context=recall.eligible_for_context,
                factual_cell_ids=recall.context_cell_ids,
                control_cell_ids=control_ids,
                support_episode_ids=tuple(support_episode_ids),
                conflict_episode_ids=recall.conflict_episode_ids,
                confidence=semantic.confidence,
                importance=semantic.importance,
            ))
        groups.sort(key=lambda group: group.group_id)
        packet_id = self._packet_id(query, groups)
        return RosetteContextPacket(packet_id=packet_id, query=query, groups=tuple(groups))

    @staticmethod
    def _packet_id(query: str, groups: list[RosetteContextGroup]) -> str:
        payload = {"query": query, "groups": [group.to_dict() for group in groups]}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"rosette-{digest[:20]}"


def _bind_replayed_episode(store: OrionMemoryStore, binder: TrifidEpisodeBinder, *, scene: str, stamp: float, outcome: str) -> EpisodeFrame:
    source = store.write_cell(memory_system=MemorySystem.EPISODIC, content=f"robot {scene} sample", summary=f"{scene} sample", source="obs", time_index=stamp)
    frame = binder.bind(store, source_cell_ids=[source.cell_id], scene=scene, entities=["robot"], goal="collect sample", action="inspect", outcome=outcome)
    completion = TrifidEpisodePatternCompleter().complete_frame(store, frame, query=f"robot {scene}", time_index=stamp, window_seconds=1.0)
    assert completion is not None
    TrifidEpisodeReplayer.record_completion(store, completion)
    return frame


def _create_lagoon_schema(
    store: OrionMemoryStore,
    binder: TrifidEpisodeBinder,
    *,
    scene: str,
    offset: float,
    with_conflict: bool,
    approved: bool,
) -> tuple[str, EpisodeFrame | None, str | None]:
    _bind_replayed_episode(store, binder, scene=scene, stamp=offset + 100.0, outcome="secured")
    _bind_replayed_episode(store, binder, scene=scene, stamp=offset + 130.0, outcome="secured")
    cluster = next(cluster for cluster in LagoonConsolidationClusterer().cluster(store, binder, max_gap_seconds=50.0) if cluster.signature[0] == scene)
    candidate = LagoonSchemaCandidateBuilder().build([cluster])[0]
    semantic = LagoonApprovedConsolidator().consolidate(store, cluster, candidate, approval_id=f"rosette-{scene}")
    if not with_conflict:
        return semantic.semantic_cell_id, None, None
    conflict = _bind_replayed_episode(store, binder, scene=scene, stamp=offset + 200.0, outcome="failed")
    LagoonSchemaConflictGuard().detect_and_mark(store, conflict)
    if not approved:
        return semantic.semantic_cell_id, conflict, None
    conflict_link = next(link for link in store.links if link.source_cell_id == conflict.anchor_cell_id and link.target_cell_id == semantic.semantic_cell_id and link.link_type.value == "conflict")
    review = LagoonConflictReviewBuilder().build(store, binder, conflict_link)
    assert review is not None
    decision = LagoonConflictDecisionLedger().decide(store, review, approval_id=f"rosette-{scene}-approved")
    return semantic.semantic_cell_id, conflict, decision.decision_cell_id


def run_stage139_rosette_context_packet_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    semantic_id, conflict, decision_id = _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    packet = RosetteContextAssembler().assemble(store, binder, "laboratory collect")
    group = packet.groups[0] if packet.groups else None
    summary = {
        "stage": "stage139_rosette_context_packet",
        "packet": packet.to_dict(),
        "stage_gates": {
            "approved_group_assembled": group is not None and group.status == "preserve_both" and group.eligible_for_context,
            "factual_cells_are_separate": group is not None and group.factual_cell_ids == (semantic_id, conflict.anchor_cell_id if conflict else ""),
            "decision_is_control_only": group is not None and group.control_cell_ids == (decision_id,) and decision_id not in group.factual_cell_ids,
            "support_provenance_preserved": group is not None and len(group.support_episode_ids) == 2,
            "packet_id_is_deterministic": packet.packet_id == RosetteContextAssembler().assemble(store, binder, "laboratory collect").packet_id,
            "assembly_read_traced": any(event.details.get("rosette_context_packet") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
