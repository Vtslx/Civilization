from __future__ import annotations

import argparse
import json

from .stage44a_local_multiclass_integration import run_qwen3_stage44a_local_multiclass_integration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage44A local multiclass path integration")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", default="202")
    parser.add_argument("--samples-per-label", type=int, default=24)
    parser.add_argument("--train-groups", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--preferred-device", default="cuda")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    kwargs = {}
    if args.smoke:
        kwargs.update(
            samples_per_label=4,
            train_groups=2,
            max_length=64,
            strict_stage_gates=False,
            memory_steps=2,
            rule_steps=2,
            state_steps=2,
            conflict_steps=2,
            combined_steps=2,
            full_hidden_steps=2,
        )
    else:
        kwargs.update(
            samples_per_label=args.samples_per_label,
            train_groups=args.train_groups,
            max_length=args.max_length,
            strict_stage_gates=True,
        )
    summary = run_qwen3_stage44a_local_multiclass_integration(
        seeds=seeds,
        preferred_device=args.preferred_device,
        **kwargs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
