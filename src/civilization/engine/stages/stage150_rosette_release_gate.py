from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .stage144_rosette_release_gate import run_stage144_rosette_release_gate
from .stage145_rosette_honeycomb_graph import run_stage145_rosette_honeycomb_graph_smoke
from .stage146_rosette_honeycomb_traversal import run_stage146_rosette_honeycomb_traversal_smoke
from .stage147_rosette_honeycomb_validator import run_stage147_rosette_honeycomb_validator_smoke
from .stage148_rosette_honeycomb_snapshot import run_stage148_rosette_honeycomb_snapshot_smoke
from .stage149_rosette_honeycomb_recovery import run_stage149_rosette_honeycomb_recovery_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage150_rosette_release_gate")


_GRAPH_RUNNERS: tuple[tuple[str, Callable[..., dict]], ...] = (
    ("stage145", run_stage145_rosette_honeycomb_graph_smoke),
    ("stage146", run_stage146_rosette_honeycomb_traversal_smoke),
    ("stage147", run_stage147_rosette_honeycomb_validator_smoke),
    ("stage148", run_stage148_rosette_honeycomb_snapshot_smoke),
    ("stage149", run_stage149_rosette_honeycomb_recovery_smoke),
)


def run_stage150_rosette_release_gate(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    context_release = run_stage144_rosette_release_gate(output_dir=output / "context-stage144")
    graph_stages = {stage: runner(output_dir=output / stage) for stage, runner in _GRAPH_RUNNERS}
    upstream_lagoon = context_release.get("lagoon_release", {})
    upstream_eagle = context_release.get("eagle_release", {})
    stage_gates = {
        "lagoon_v0_00_04_green": upstream_lagoon.get("version") == "v0.00.04" and upstream_lagoon.get("passes_stage_gate") is True,
        "eagle_v0_00_05_green": upstream_eagle.get("version") == "v0.00.05" and upstream_eagle.get("passes_stage_gate") is True,
        "rosette_context_release_green": context_release.get("passes_stage_gate") is True,
        "all_honeycomb_stages_pass": all(summary.get("passes_stage_gate") is True for summary in graph_stages.values()),
        "honeycomb_stage_set_complete": set(graph_stages) == {f"stage{number}" for number in range(145, 150)},
        "all_release_artifacts_present": (output / "context-stage144" / "summary.json").exists() and context_release.get("stage_gates", {}).get("all_context_artifacts_present") is True and all(all((output / stage / filename).exists() for filename in ("summary.json", "memory_cells.json", "memory_links.json", "trace_events.jsonl")) for stage in graph_stages) and (output / "stage148" / "honeycomb.json").exists() and (output / "stage149" / "valid-honeycomb.json").exists(),
        "roadmap_contracts_covered": all(
            graph_stages[stage]["stage_gates"].get(key) is True
            for stage, key in (
                ("stage145", "multi_scale_nodes_present"),
                ("stage145", "multi_adjacency_edges_present"),
                ("stage146", "full_traversal_reaches_all_scales"),
                ("stage147", "live_topology_rechecked"),
                ("stage148", "canonical_hash_matches"),
                ("stage149", "live_topology_revalidated"),
            )
        ),
    }
    summary = {"stage": "stage150_rosette_release_gate", "version": "v0.00.06", "version_chain": ["v0.00.04", "v0.00.05", "v0.00.06"], "context_release": context_release, "honeycomb_stage_summaries": graph_stages, "stage_gates": stage_gates}
    summary["passes_stage_gate"] = all(stage_gates.values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
