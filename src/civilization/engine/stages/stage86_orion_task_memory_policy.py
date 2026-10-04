from __future__ import annotations

import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage86_orion_task_memory_policy")


class OrionTaskMemoryPolicy:
    def __init__(self, *, allow_resolved_global: bool = False) -> None:
        self.allow_resolved_global = allow_resolved_global

    def select(self, rows: list[dict]) -> dict:
        selected, excluded = [], []
        for row in rows:
            if row["tier"] == "session":
                selected.append(row["cell_id"])
            elif self.allow_resolved_global and row.get("resolution_status") == "winner":
                selected.append(row["cell_id"])
            else:
                excluded.append({"cell_id": row["cell_id"], "reason": "global_not_resolved_winner"})
        return {"selected_cell_ids": selected, "excluded": excluded, "allow_resolved_global": self.allow_resolved_global}


def run_stage86_orion_task_policy_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    rows = [
        {"tier": "session", "cell_id": "session-1", "resolution_status": "not_applicable"},
        {"tier": "global", "cell_id": "winner-1", "resolution_status": "winner"},
        {"tier": "global", "cell_id": "loser-1", "resolution_status": "loser"},
        {"tier": "global", "cell_id": "unresolved-1", "resolution_status": "unresolved"},
    ]
    disabled = OrionTaskMemoryPolicy().select(rows)
    enabled = OrionTaskMemoryPolicy(allow_resolved_global=True).select(rows)
    summary = {"stage": "stage86_orion_task_memory_policy", "disabled": disabled, "enabled": enabled, "stage_gates": {"session_always_selected": "session-1" in disabled["selected_cell_ids"] and "session-1" in enabled["selected_cell_ids"], "winner_requires_opt_in": "winner-1" not in disabled["selected_cell_ids"] and "winner-1" in enabled["selected_cell_ids"], "unsafe_global_excluded": all(cell not in enabled["selected_cell_ids"] for cell in ("loser-1", "unresolved-1"))}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
