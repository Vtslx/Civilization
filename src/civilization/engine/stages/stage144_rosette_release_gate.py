from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .stage132_lagoon_release_gate import run_stage132_lagoon_release_gate
from .stage132_lagoon_release_gate import run_stage132_lagoon_release_gate
from .stage138_eagle_release_gate import run_stage138_eagle_release_gate
from .stage139_rosette_context_packet import run_stage139_rosette_context_packet_smoke
from .stage140_rosette_atomic_budget import run_stage140_rosette_atomic_budget_smoke
from .stage141_rosette_packet_validator import run_stage141_rosette_packet_validator_smoke
from .stage142_rosette_packet_snapshot import run_stage142_rosette_packet_snapshot_smoke
from .stage143_rosette_snapshot_recovery import run_stage143_rosette_snapshot_recovery_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage144_rosette_release_gate")


_ROSETTE_RUNNERS: tuple[tuple[str, Callable[..., dict]], ...] = (
    ("stage139", run_stage139_rosette_context_packet_smoke),
    ("stage140", run_stage140_rosette_atomic_budget_smoke),
    ("stage141", run_stage141_rosette_packet_validator_smoke),
    ("stage142", run_stage142_rosette_packet_snapshot_smoke),
    ("stage143", run_stage143_rosette_snapshot_recovery_smoke),
)


def run_stage144_rosette_release_gate(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lagoon = run_stage132_lagoon_release_gate(output_dir=output / "lagoon-stage132")
    eagle = run_stage138_eagle_release_gate(output_dir=output / "eagle-stage138")
    rosette = {stage: runner(output_dir=output / stage) for stage, runner in _ROSETTE_RUNNERS}
    contract_keys = {
        "stage139": "decision_is_control_only",
        "stage140": "preserve_both_group_is_atomic",
        "stage141": "incomplete_pair_rejected",
        "stage142": "canonical_hash_matches",
        "stage143": "tampered_snapshot_rejected",
    }
    stage_gates = {
        "lagoon_release_remains_green": lagoon.get("passes_stage_gate") is True,
        "eagle_release_remains_green": eagle.get("passes_stage_gate") is True,
        "all_rosette_stages_pass": all(summary.get("passes_stage_gate") is True for summary in rosette.values()),
        "all_release_summaries_present": (output / "lagoon-stage132" / "summary.json").exists() and (output / "eagle-stage138" / "summary.json").exists() and all((output / stage / "summary.json").exists() for stage in rosette),
        "all_context_artifacts_present": all(all((output / stage / filename).exists() for filename in ("memory_cells.json", "memory_links.json", "trace_events.jsonl")) for stage in rosette) and (output / "stage142" / "packet.json").exists() and (output / "stage143" / "valid-packet.json").exists(),
        "rosette_stage_set_complete": set(rosette) == {f"stage{number}" for number in range(139, 144)},
        "critical_contracts_covered": all(rosette[stage]["stage_gates"].get(key) is True for stage, key in contract_keys.items()),
    }
    summary = {
        "stage": "stage144_rosette_release_gate",
        "version": "v0.00.06",
        "lagoon_release": lagoon,
        "eagle_release": eagle,
        "rosette_stage_summaries": rosette,
        "stage_gates": stage_gates,
    }
    summary["passes_stage_gate"] = all(stage_gates.values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
