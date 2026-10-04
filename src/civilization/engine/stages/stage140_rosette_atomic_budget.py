from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import RosetteContextAssembler, RosetteContextGroup, RosetteContextPacket, _create_lagoon_schema
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage140_rosette_atomic_budget")


@dataclass(frozen=True)
class RosetteRoutedPacket:
    source_packet_id: str
    query: str
    max_fact_cells: int
    used_fact_cells: int
    selected_groups: tuple[RosetteContextGroup, ...]
    dropped_group_ids: tuple[str, ...]

    @property
    def factual_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell_id for group in self.selected_groups for cell_id in group.factual_cell_ids)

    @property
    def control_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell_id for group in self.selected_groups for cell_id in group.control_cell_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_groups"] = [group.to_dict() for group in self.selected_groups]
        return payload


class RosetteAtomicBudgetRouter:
    """Routes whole Rosette groups under a deterministic factual-cell budget."""

    def route(self, packet: RosetteContextPacket, *, max_fact_cells: int) -> RosetteRoutedPacket:
        if max_fact_cells <= 0:
            raise ValueError("max_fact_cells must be positive")
        status_rank = {"stable": 0, "preserve_both": 1, "needs_review": 2}
        ordered = sorted(packet.groups, key=lambda group: (
            not group.eligible_for_context,
            status_rank.get(group.status, 99),
            -group.importance,
            -group.confidence,
            group.group_id,
        ))
        selected: list[RosetteContextGroup] = []
        dropped: list[str] = []
        used = 0
        for group in ordered:
            group_cost = len(group.factual_cell_ids)
            if not group.eligible_for_context or group_cost == 0 or used + group_cost > max_fact_cells:
                dropped.append(group.group_id)
                continue
            selected.append(group)
            used += group_cost
        return RosetteRoutedPacket(packet.packet_id, packet.query, max_fact_cells, used, tuple(selected), tuple(dropped))


def run_stage140_rosette_atomic_budget_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    lab_semantic, lab_conflict, _decision = _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    clinic_semantic, _none, _none_decision = _create_lagoon_schema(store, binder, scene="clinic", offset=1000.0, with_conflict=False, approved=False)
    pending_semantic, _pending_conflict, _pending_decision = _create_lagoon_schema(store, binder, scene="warehouse", offset=2000.0, with_conflict=True, approved=False)
    packet = RosetteContextAssembler().assemble(store, binder, "robot collect")
    router = RosetteAtomicBudgetRouter()
    tight = router.route(packet, max_fact_cells=2)
    full = router.route(packet, max_fact_cells=3)
    lab_group = next(group for group in packet.groups if group.factual_cell_ids[0] == lab_semantic)
    clinic_group = next(group for group in packet.groups if group.factual_cell_ids[0] == clinic_semantic)
    pending_group = next(group for group in packet.groups if group.factual_cell_ids[0] == pending_semantic)
    summary = {
        "stage": "stage140_rosette_atomic_budget",
        "tight": tight.to_dict(),
        "full": full.to_dict(),
        "stage_gates": {
            "stable_group_selected_first": tight.factual_cell_ids == (clinic_semantic,),
            "preserve_both_group_is_atomic": lab_group.group_id in tight.dropped_group_ids and (lab_conflict is not None and lab_conflict.anchor_cell_id not in tight.factual_cell_ids),
            "full_budget_keeps_complete_pair": set(full.factual_cell_ids) == {clinic_semantic, lab_semantic, lab_conflict.anchor_cell_id if lab_conflict else ""},
            "control_follows_selected_group": len(full.control_cell_ids) == 1 and tight.control_cell_ids == (),
            "unapproved_group_always_dropped": pending_group.group_id in tight.dropped_group_ids and pending_group.group_id in full.dropped_group_ids,
            "budget_accounting_exact": tight.used_fact_cells == 1 and full.used_fact_cells == 3,
            "routing_is_deterministic": tight == router.route(packet, max_fact_cells=2),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
