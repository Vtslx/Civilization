from __future__ import annotations

import json

from .full_hidden_centroid_alignment import run_qwen3_full_hidden_centroid_alignment


def main() -> None:
    summary = run_qwen3_full_hidden_centroid_alignment()
    print(
        "qwen3_full_hidden_centroid_alignment_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_multiclass_repair_planning": summary["allows_multiclass_repair_planning"],
                "fixed_centroid_average_after": summary["fixed_centroid_average_after"],
                "final_projected_accuracy": summary["final_projected_accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
