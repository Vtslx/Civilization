from __future__ import annotations

import json

from .multiclass_group_curriculum_repair import run_qwen3_multiclass_group_curriculum_repair


def main() -> None:
    summary = run_qwen3_multiclass_group_curriculum_repair()
    print(
        "qwen3_multiclass_group_curriculum_repair_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_stage27b_rerun": summary["allows_stage27b_rerun"],
                "local_projected_accuracy": summary["local_projected_accuracy"],
                "local_fixed_centroid_accuracy": summary["local_fixed_centroid_accuracy"],
                "group_success": summary["group_success"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

