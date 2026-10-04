from __future__ import annotations

import argparse
import json

from .rule_conflict_group_recovery import run_qwen3_rule_conflict_group_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 41 Rule/Conflict/Combined five-candidate recovery.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preferred-device", default=None, choices=(None, "auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    kwargs = {"preferred_device": args.preferred_device}
    if args.smoke:
        kwargs.update(
            local_samples_per_label=4,
            local_train_groups=2,
            rule_steps=4,
            conflict_steps=4,
            combined_steps=4,
            projector_sanity_steps=4,
            strict_stage_gates=False,
        )
    summary = run_qwen3_rule_conflict_group_recovery(**kwargs)
    print(
        "qwen3_rule_conflict_group_recovery_complete "
        + json.dumps(
            {
                "passes_stage_gate": summary["passes_stage_gate"],
                "allows_stage42": summary["allows_stage42"],
                "completed_stages": summary.get("completed_stages", []),
                "failed_stage": summary.get("failed_stage"),
                "failed_gate": summary.get("failed_gate"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
