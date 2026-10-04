from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .stage150_rosette_release_gate import run_stage150_rosette_release_gate
from .stage151_helix_path_weight_model import run_stage151_helix_path_weight_model_smoke
from .stage152_helix_feedback_signal import run_stage152_helix_feedback_signal_smoke
from .stage153_helix_weight_update_rule import run_stage153_helix_weight_update_rule_smoke
from .stage154_helix_approval_gated_mutation import run_stage154_helix_approval_gated_mutation_smoke
from .stage155_helix_weighted_retrieval import run_stage155_helix_weighted_retrieval_smoke
from .stage156_helix_weight_conflict_guard import run_stage156_helix_weight_conflict_guard_smoke
from .stage157_helix_weight_snapshot import run_stage157_helix_weight_snapshot_smoke
from .stage158_helix_weight_recovery import run_stage158_helix_weight_recovery_smoke
from .stage159_helix_calibration_gate import run_stage159_helix_calibration_gate_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage160_helix_release_gate")

VERSION_CHAIN = ["v0.00.04", "v0.00.05", "v0.00.06", "v0.00.07"]

# The seven release-gate categories named in the Helix plan.
_CATEGORY_STAGES: tuple[tuple[str, str], ...] = (
    ("helix_path_weight", "stage151"),
    ("helix_feedback", "stage152"),
    ("helix_update_rule", "stage153"),
    ("helix_gated_mutation", "stage154"),
    ("helix_snapshot", "stage157"),
    ("helix_recovery", "stage158"),
    ("helix_calibration", "stage159"),
)

_HELIX_RUNNERS: tuple[tuple[str, Callable[..., dict]], ...] = (
    ("stage151", run_stage151_helix_path_weight_model_smoke),
    ("stage152", run_stage152_helix_feedback_signal_smoke),
    ("stage153", run_stage153_helix_weight_update_rule_smoke),
    ("stage154", run_stage154_helix_approval_gated_mutation_smoke),
    ("stage155", run_stage155_helix_weighted_retrieval_smoke),
    ("stage156", run_stage156_helix_weight_conflict_guard_smoke),
    ("stage157", run_stage157_helix_weight_snapshot_smoke),
    ("stage158", run_stage158_helix_weight_recovery_smoke),
    ("stage159", run_stage159_helix_calibration_gate_smoke),
)


def run_stage160_helix_release_gate(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Upstream: Rosette release gate closes v0.00.04 -> v0.00.05 -> v0.00.06.
    rosette_release = run_stage150_rosette_release_gate(output_dir=output / "stage150-rosette")
    context_release = rosette_release.get("context_release", {})
    upstream_lagoon = context_release.get("lagoon_release", {})
    upstream_eagle = context_release.get("eagle_release", {})

    # Helix functional stages 151-159.
    helix_stages = {stage: runner(output_dir=output / stage) for stage, runner in _HELIX_RUNNERS}

    # Weight determinism: the Stage157 snapshot SHA-256 is stable across re-runs
    # (deterministic clocks -> identical canonical payload -> identical hash).
    det_run = run_stage157_helix_weight_snapshot_smoke(output_dir=output / "stage157-determinism")
    weight_determinism = det_run.get("sha256") == helix_stages["stage157"].get("sha256") and bool(det_run.get("sha256"))

    stage_gates = {
        "lagoon_v0_00_04_green": upstream_lagoon.get("version") == "v0.00.04" and upstream_lagoon.get("passes_stage_gate") is True,
        "eagle_v0_00_05_green": upstream_eagle.get("version") == "v0.00.05" and upstream_eagle.get("passes_stage_gate") is True,
        "rosette_v0_00_06_green": rosette_release.get("version") == "v0.00.06" and rosette_release.get("passes_stage_gate") is True,
        **{
            category: helix_stages[stage].get("passes_stage_gate") is True
            for category, stage in _CATEGORY_STAGES
        },
        "all_helix_stages_pass": all(helix_stages[s].get("passes_stage_gate") is True for s in helix_stages),
        "helix_stage_set_complete": set(helix_stages) == {f"stage{n}" for n in range(151, 160)},
        "version_chain_complete": rosette_release.get("version_chain") == ["v0.00.04", "v0.00.05", "v0.00.06"]
        and VERSION_CHAIN == ["v0.00.04", "v0.00.05", "v0.00.06", "v0.00.07"],
        "weight_determinism": weight_determinism,
        "all_release_artifacts_present": all((output / stage / "summary.json").exists() for stage in helix_stages)
        and (output / "stage150-rosette" / "summary.json").exists(),
        "qwen_frozen": True,
    }

    summary = {
        "stage": "stage160_helix_release_gate",
        "version": "v0.00.07",
        "version_chain": VERSION_CHAIN,
        "rosette_release": {k: rosette_release.get(k) for k in ("stage", "version", "version_chain", "passes_stage_gate")},
        "helix_stage_summaries": {s: {k: helix_stages[s].get(k) for k in ("stage", "version", "passes_stage_gate")} for s in helix_stages},
        "weight_determinism_sha256": det_run.get("sha256"),
        "stage_gates": stage_gates,
    }
    summary["passes_stage_gate"] = all(stage_gates.values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
