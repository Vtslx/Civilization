from __future__ import annotations

import argparse
import json

from .stage44c_real_task_runner import run_qwen3_stage44c_real_task_runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage44C unified local + external real-task runner")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--allow-dataset-download", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    summary = run_qwen3_stage44c_real_task_runner(
        seeds=seeds,
        preferred_device=args.preferred_device,
        smoke=args.smoke,
        reuse_existing=args.reuse_existing,
        allow_dataset_download=args.allow_dataset_download,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
