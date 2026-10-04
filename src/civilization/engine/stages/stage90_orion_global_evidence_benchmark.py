from __future__ import annotations

import json
from pathlib import Path

from .stage86_orion_task_memory_policy import OrionTaskMemoryPolicy


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage90_orion_global_evidence_benchmark")


def run_stage90_orion_global_evidence_benchmark(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    rows = [{"tier": "session", "cell_id": "session:local", "resolution_status": "not_applicable"}, {"tier": "global", "cell_id": "global:winner", "resolution_status": "winner"}]
    conditions = {
        "session_only": OrionTaskMemoryPolicy().select(rows[:1]),
        "global_denied": OrionTaskMemoryPolicy().select(rows),
        "global_allowed": OrionTaskMemoryPolicy(allow_resolved_global=True).select(rows),
    }
    summary = {"stage": "stage90_orion_global_evidence_benchmark", "conditions": conditions, "stage_gates": {"session_only_isolated": conditions["session_only"]["selected_cell_ids"] == ["session:local"], "denied_excludes_winner": "global:winner" not in conditions["global_denied"]["selected_cell_ids"], "allowed_injects_winner": "global:winner" in conditions["global_allowed"]["selected_cell_ids"]}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
