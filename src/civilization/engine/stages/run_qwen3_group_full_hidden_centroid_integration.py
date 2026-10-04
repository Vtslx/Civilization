from __future__ import annotations

import argparse
import json

from .group_full_hidden_centroid_integration import run_qwen3_group_full_hidden_centroid_integration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 42 group full-hidden fixed-centroid integration.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    kwargs = {"preferred_device": args.preferred_device}
    if args.smoke:
        kwargs.update(
            local_samples_per_label=4,
            local_train_groups=2,
            alignment_steps=4,
            strict_stage_gates=False,
            output_dir="artifacts/civilization/group_full_hidden_centroid_integration_smoke",
        )
    summary = run_qwen3_group_full_hidden_centroid_integration(**kwargs)
    print(
        "qwen3_group_full_hidden_centroid_integration_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_stage43": summary["allows_stage43"],
                "fixed_centroid_average_after": summary["fixed_centroid_average_after"],
                "failed_stage": summary.get("failed_stage"),
                "failed_gate": summary.get("failed_gate"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
