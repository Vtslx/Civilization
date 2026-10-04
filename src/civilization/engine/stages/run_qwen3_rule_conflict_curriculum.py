from __future__ import annotations

import json

from .rule_conflict_curriculum import run_qwen3_rule_conflict_curriculum


def main() -> None:
    summary = run_qwen3_rule_conflict_curriculum()
    print(
        "qwen3_rule_conflict_curriculum_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_multiclass_repair_planning": summary["allows_multiclass_repair_planning"],
                "final_projected_accuracy": summary["final_projected_accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
