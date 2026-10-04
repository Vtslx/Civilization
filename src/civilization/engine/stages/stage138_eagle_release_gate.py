from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .stage132_lagoon_release_gate import run_stage132_lagoon_release_gate
from .stage133_eagle_task_trace import run_stage133_eagle_task_trace_smoke
from .stage134_eagle_skill_candidates import run_stage134_eagle_skill_candidate_smoke
from .stage135_eagle_approved_skill import run_stage135_eagle_approved_skill_smoke
from .stage136_eagle_skill_retrieval import run_stage136_eagle_skill_retrieval_smoke
from .stage137_eagle_skill_conflict_guard import run_stage137_eagle_skill_conflict_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage138_eagle_release_gate")


_EAGLE_RUNNERS: tuple[tuple[str, Callable[..., dict]], ...] = (
    ("stage133", run_stage133_eagle_task_trace_smoke),
    ("stage134", run_stage134_eagle_skill_candidate_smoke),
    ("stage135", run_stage135_eagle_approved_skill_smoke),
    ("stage136", run_stage136_eagle_skill_retrieval_smoke),
    ("stage137", run_stage137_eagle_skill_conflict_smoke),
)


def run_stage138_eagle_release_gate(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lagoon = run_stage132_lagoon_release_gate(output_dir=output / "lagoon-stage132")
    stages = {stage: runner(output_dir=output / stage) for stage, runner in _EAGLE_RUNNERS}
    stage_gates = {
        "lagoon_release_remains_green": lagoon.get("passes_stage_gate") is True,
        "all_eagle_stages_pass": all(summary.get("passes_stage_gate") is True for summary in stages.values()),
        "stage_set_complete": set(stages) == {f"stage{number}" for number in range(133, 138)},
        "all_summaries_present": (output / "lagoon-stage132" / "summary.json").exists() and all((output / stage / "summary.json").exists() for stage in stages),
        "evidence_artifacts_complete": (output / "stage133" / "task_traces.json").exists() and (output / "stage134" / "skill_candidates.json").exists() and all(all((output / stage / filename).exists() for filename in ("memory_cells.json", "memory_links.json", "trace_events.jsonl")) for stage in ("stage135", "stage136", "stage137")),
        "procedural_contracts_covered": all(
            stages[stage]["stage_gates"].get(key) is True
            for stage, key in (
                ("stage133", "success_failure_preserved"),
                ("stage134", "failure_evidence_preserved"),
                ("stage135", "procedural_skill_created"),
                ("stage136", "trace_links_verified"),
                ("stage137", "divergent_failure_conflicted"),
            )
        ),
    }
    summary = {"stage": "stage138_eagle_release_gate", "version": "v0.00.05", "lagoon_release": lagoon, "eagle_stage_summaries": stages, "stage_gates": stage_gates}
    summary["passes_stage_gate"] = all(stage_gates.values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
