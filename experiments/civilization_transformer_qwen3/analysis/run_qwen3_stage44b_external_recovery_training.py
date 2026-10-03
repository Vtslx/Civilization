from __future__ import annotations

import argparse
import json

from .stage44b_external_recovery_training import (
    run_qwen3_stage44b_external_recovery_training,
    run_qwen3_stage44b1_external_recovery_training,
    run_qwen3_stage44b2_external_recovery_training,
    run_qwen3_stage44b3_external_recovery_training,
    run_qwen3_stage44b4_external_recovery_training,
    run_qwen3_stage44b5_external_recovery_training,
    run_qwen3_stage44b6_external_recovery_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage44B seed 202 external recovery training")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--stage44b1", action="store_true")
    parser.add_argument("--stage44b2", action="store_true")
    parser.add_argument("--stage44b3", action="store_true")
    parser.add_argument("--stage44b4", action="store_true")
    parser.add_argument("--stage44b5", action="store_true")
    parser.add_argument("--stage44b6", action="store_true")
    parser.add_argument("--allow-dataset-download", action="store_true")
    args = parser.parse_args()
    kwargs = {}
    if args.model_path:
        kwargs["model_path"] = args.model_path
    if args.smoke:
        kwargs.update(task_steps=2, combined_steps=2, train_per_label=2, heldout_per_label=2, strict_stage_gates=False)
    if sum(bool(value) for value in (args.stage44b1, args.stage44b2, args.stage44b3, args.stage44b4, args.stage44b5, args.stage44b6)) > 1:
        raise SystemExit("--stage44b1, --stage44b2, --stage44b3, --stage44b4, --stage44b5, and --stage44b6 are mutually exclusive")
    if args.stage44b6:
        summary = run_qwen3_stage44b6_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            allow_dataset_download=args.allow_dataset_download,
            **kwargs,
        )
    elif args.stage44b5:
        summary = run_qwen3_stage44b5_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            **kwargs,
        )
    elif args.stage44b4:
        summary = run_qwen3_stage44b4_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            **kwargs,
        )
    elif args.stage44b3:
        summary = run_qwen3_stage44b3_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            **kwargs,
        )
    elif args.stage44b2:
        summary = run_qwen3_stage44b2_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            **kwargs,
        )
    elif args.stage44b1:
        summary = run_qwen3_stage44b1_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            **kwargs,
        )
    else:
        summary = run_qwen3_stage44b_external_recovery_training(
            seed=args.seed,
            preferred_device=args.preferred_device,
            evaluation_only=args.evaluation_only,
            **kwargs,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
