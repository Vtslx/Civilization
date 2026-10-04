from __future__ import annotations

import json

from .projected_binary_path_necessity import run_qwen3_projected_binary_path_necessity


def main() -> None:
    summary = run_qwen3_projected_binary_path_necessity()
    print(
        "qwen3_projected_binary_path_necessity_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_multiclass_repair_planning": summary["allows_multiclass_repair_planning"],
                "projected_answer_accuracy": summary["projected_answer_accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
