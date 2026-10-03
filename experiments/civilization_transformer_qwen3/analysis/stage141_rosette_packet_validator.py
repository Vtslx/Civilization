from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import RosetteContextAssembler, RosetteContextGroup, RosetteContextPacket, _create_lagoon_schema
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage141_rosette_packet_validator")


@dataclass(frozen=True)
class RosettePacketValidationReport:
    valid: bool
    errors: tuple[str, ...]
    checked_group_ids: tuple[str, ...]


class RosettePacketValidator:
    """Fail-closed validation for Rosette packet references, provenance, and control links."""

    def validate(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, packet: RosetteContextPacket) -> RosettePacketValidationReport:
        errors: list[str] = []
        group_ids = [group.group_id for group in packet.groups]
        if len(group_ids) != len(set(group_ids)):
            errors.append("duplicate_group_id")
        expected_packet_id = RosetteContextAssembler._packet_id(packet.query, list(packet.groups))
        if packet.packet_id != expected_packet_id:
            errors.append("packet_id_mismatch")
        factual_ids = list(packet.factual_cell_ids)
        control_ids = list(packet.control_cell_ids)
        if len(factual_ids) != len(set(factual_ids)):
            errors.append("duplicate_factual_cell")
        if set(factual_ids) & set(control_ids):
            errors.append("fact_control_overlap")
        evidence = store.read_cells(
            [*factual_ids, *control_ids],
            trace_details={"rosette_packet_validation": True, "packet_id": packet.packet_id},
        )
        if len(evidence) != len(factual_ids) + len(control_ids):
            errors.append("missing_or_inactive_reference")
        frame_by_id = binder.frames
        for group in packet.groups:
            self._validate_group(store, frame_by_id, group, errors)
        return RosettePacketValidationReport(not errors, tuple(errors), tuple(group_ids))

    @staticmethod
    def _validate_group(store: OrionMemoryStore, frames: dict, group: RosetteContextGroup, errors: list[str]) -> None:
        if not group.factual_cell_ids:
            errors.append(f"{group.group_id}:empty_facts")
            return
        semantic = store.cells.get(group.factual_cell_ids[0])
        if semantic is None or semantic.memory_system != MemorySystem.SEMANTIC or semantic.source != "lagoon_stage124":
            errors.append(f"{group.group_id}:invalid_semantic_fact")
            return
        for cell_id in group.factual_cell_ids[1:]:
            cell = store.cells.get(cell_id)
            if cell is None or cell.memory_system != MemorySystem.EPISODIC:
                errors.append(f"{group.group_id}:invalid_episodic_fact")
        expected_support = semantic.metadata.get("lagoon_support_episode_ids")
        expected_anchors = semantic.metadata.get("consolidated_from")
        if not isinstance(expected_support, list) or tuple(expected_support) != group.support_episode_ids:
            errors.append(f"{group.group_id}:support_provenance_mismatch")
        elif not isinstance(expected_anchors, list) or [frames[item].anchor_cell_id for item in expected_support if item in frames] != expected_anchors:
            errors.append(f"{group.group_id}:support_anchor_mismatch")
        conflict_anchors = [frames[item].anchor_cell_id for item in group.conflict_episode_ids if item in frames]
        if len(conflict_anchors) != len(group.conflict_episode_ids):
            errors.append(f"{group.group_id}:missing_conflict_episode")
        actual_conflict_anchors = {
            link.source_cell_id for link in store.links
            if link.link_type == MemoryLinkType.CONFLICT and link.target_cell_id == semantic.cell_id and link.metadata.get("reason") == "lagoon_outcome_conflict"
        }
        if set(conflict_anchors) != actual_conflict_anchors:
            errors.append(f"{group.group_id}:conflict_link_mismatch")
        if group.status == "stable":
            if not group.eligible_for_context or len(group.factual_cell_ids) != 1 or group.control_cell_ids or group.conflict_episode_ids:
                errors.append(f"{group.group_id}:invalid_stable_group")
        elif group.status == "needs_review":
            if group.eligible_for_context or len(group.factual_cell_ids) != 1 or group.control_cell_ids or not group.conflict_episode_ids:
                errors.append(f"{group.group_id}:invalid_review_group")
        elif group.status == "preserve_both":
            expected_facts = {semantic.cell_id, *conflict_anchors}
            if not group.eligible_for_context or set(group.factual_cell_ids) != expected_facts or len(group.control_cell_ids) != len(conflict_anchors):
                errors.append(f"{group.group_id}:incomplete_preserve_both_group")
            RosettePacketValidator._validate_decisions(store, semantic.cell_id, conflict_anchors, group.control_cell_ids, group.group_id, errors)
        else:
            errors.append(f"{group.group_id}:unknown_status")

    @staticmethod
    def _validate_decisions(store: OrionMemoryStore, semantic_cell_id: str, conflict_anchors: list[str], control_ids: tuple[str, ...], group_id: str, errors: list[str]) -> None:
        covered_anchors: set[str] = set()
        for control_id in control_ids:
            decision = store.cells.get(control_id)
            if decision is None or decision.memory_system != MemorySystem.PROCEDURAL or decision.source != "lagoon_stage130" or decision.metadata.get("lagoon_resolution") != "preserve_both":
                errors.append(f"{group_id}:invalid_decision_control")
                continue
            fingerprint = decision.metadata.get("lagoon_conflict_fingerprint")
            for anchor in conflict_anchors:
                expected = f"lagoon-conflict:{anchor}|{semantic_cell_id}|preserve_both"
                if fingerprint != expected:
                    continue
                task_sources = {
                    link.source_cell_id for link in store.links
                    if link.link_type == MemoryLinkType.TASK and link.target_cell_id == control_id and link.metadata.get("lagoon_conflict_resolution") == "preserve_both"
                }
                if {anchor, semantic_cell_id}.issubset(task_sources):
                    covered_anchors.add(anchor)
        if covered_anchors != set(conflict_anchors):
            errors.append(f"{group_id}:decision_coverage_mismatch")


def _repack(query: str, groups: tuple[RosetteContextGroup, ...]) -> RosetteContextPacket:
    return RosetteContextPacket(RosetteContextAssembler._packet_id(query, list(groups)), query, groups)


def run_stage141_rosette_packet_validator_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _semantic, conflict, decision_id = _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    packet = RosetteContextAssembler().assemble(store, binder, "laboratory collect")
    validator = RosettePacketValidator()
    cell_count, link_count = len(store.cells), len(store.links)
    valid = validator.validate(store, binder, packet)
    group = packet.groups[0]
    type_confused_group = replace(group, factual_cell_ids=(decision_id, *group.factual_cell_ids[1:]))
    type_confused = validator.validate(store, binder, _repack(packet.query, (type_confused_group,)))
    dangling_group = replace(group, factual_cell_ids=(*group.factual_cell_ids, "episodic-999999"))
    dangling = validator.validate(store, binder, _repack(packet.query, (dangling_group,)))
    incomplete_group = replace(group, factual_cell_ids=(group.factual_cell_ids[0],))
    incomplete = validator.validate(store, binder, _repack(packet.query, (incomplete_group,)))
    summary = {
        "stage": "stage141_rosette_packet_validator",
        "stage_gates": {
            "valid_packet_accepted": valid.valid and valid.errors == (),
            "procedural_fact_rejected": not type_confused.valid and any("invalid_semantic_fact" in error for error in type_confused.errors),
            "dangling_reference_rejected": not dangling.valid and "missing_or_inactive_reference" in dangling.errors,
            "incomplete_pair_rejected": not incomplete.valid and any("incomplete_preserve_both_group" in error for error in incomplete.errors),
            "validator_is_read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "validation_is_traced": any(event.details.get("rosette_packet_validation") for event in store.trace_events),
            "conflict_anchor_preserved": conflict is not None and conflict.anchor_cell_id in store.cells,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
