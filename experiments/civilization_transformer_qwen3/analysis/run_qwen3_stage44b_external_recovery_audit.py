from __future__ import annotations

import argparse
import json

from .stage44b_external_recovery_audit import run_qwen3_stage44b_external_recovery_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage44B external task recovery inputs")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--tasks", default="glue_rte,super_glue_cb,boolq")
    parser.add_argument("--train-per-label", type=int, default=12)
    parser.add_argument("--heldout-per-label", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    kwargs = {}
    if args.model_path:
        kwargs["model_path"] = args.model_path
    summary = run_qwen3_stage44b_external_recovery_audit(
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        task_names=tuple(value for value in args.tasks.split(",") if value),
        train_per_label=args.train_per_label,
        heldout_per_label=args.heldout_per_label,
        max_length=args.max_length,
        **kwargs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

