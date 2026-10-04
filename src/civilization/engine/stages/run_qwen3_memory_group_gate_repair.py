from __future__ import annotations

import argparse
import json

from .memory_group_gate_repair import run_qwen3_memory_group_gate_repair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 39 memory five-candidate group gate repair.")
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke configuration.")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.smoke:
        summary = run_qwen3_memory_group_gate_repair(
            seed=202,
            local_samples_per_label=4,
            local_train_groups=2,
            memory_stage_steps=4,
            full_hidden_alignment_steps=2,
            preferred_device=args.preferred_device,
            strict_stage_gates=False,
        )
    else:
        summary = run_qwen3_memory_group_gate_repair(preferred_device=args.preferred_device)
    print(
        "qwen3_memory_group_gate_repair_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_stage40": summary["allows_stage40"],
                "memory_group_projected_accuracy": summary["memory_group_projected_accuracy"],
                "memory_necessity_group_success": summary["memory_necessity_group_success"],
                "no_memory_drop": summary["no_memory_drop"],
                "failed_gate": summary["failed_gate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
