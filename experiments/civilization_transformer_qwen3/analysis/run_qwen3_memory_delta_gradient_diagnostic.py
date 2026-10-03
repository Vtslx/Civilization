from __future__ import annotations

import argparse
import json

from .memory_delta_gradient_diagnostic import run_qwen3_memory_delta_gradient_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 40 memory delta gradient and five-candidate mapping diagnostic.")
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke configuration.")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.smoke:
        summary = run_qwen3_memory_delta_gradient_diagnostic(
            seed=202,
            local_samples_per_label=4,
            local_train_groups=2,
            repair_steps=4,
            preferred_device=args.preferred_device,
        )
    else:
        summary = run_qwen3_memory_delta_gradient_diagnostic(preferred_device=args.preferred_device)
    print(
        "qwen3_memory_delta_gradient_diagnostic_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_stage41": summary["allows_stage41"],
                "memory_delta_to_option_accuracy": summary["memory_delta_to_option_accuracy"],
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
