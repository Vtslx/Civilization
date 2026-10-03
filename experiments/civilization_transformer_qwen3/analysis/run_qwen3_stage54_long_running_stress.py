from __future__ import annotations

import argparse
import json

from .stage54_long_running_service_stress import (
    DEFAULT_OUTPUT_DIR,
    Stage54LongRunConfig,
    run_stage54_fake_long_running_stress,
    run_stage54_real_long_running_stress,
)


def _controls(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage54 long-running resident service stress")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--request-interval-seconds", type=float, default=None)
    parser.add_argument("--reload-interval-seconds", type=float, default=None)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=None)
    parser.add_argument("--controls", default="full,no_memory_path,adapter_disabled")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not args.smoke and not args.real:
        parser.error("one of --smoke or --real is required")
    default = Stage54LongRunConfig(
        duration_seconds=1.0,
        request_interval_seconds=0.1,
        reload_interval_seconds=0.3,
        snapshot_interval_seconds=0.2,
    ) if args.smoke else Stage54LongRunConfig()
    config = Stage54LongRunConfig(
        duration_seconds=args.duration_seconds if args.duration_seconds is not None else default.duration_seconds,
        request_interval_seconds=args.request_interval_seconds
        if args.request_interval_seconds is not None
        else default.request_interval_seconds,
        reload_interval_seconds=args.reload_interval_seconds
        if args.reload_interval_seconds is not None
        else default.reload_interval_seconds,
        snapshot_interval_seconds=args.snapshot_interval_seconds
        if args.snapshot_interval_seconds is not None
        else default.snapshot_interval_seconds,
        controls=_controls(args.controls),
        min_reload_count=1,
    )
    if args.smoke:
        summary = run_stage54_fake_long_running_stress(output_dir=args.output_dir, config=config)
    else:
        summary = run_stage54_real_long_running_stress(
            output_dir=args.output_dir,
            config=config,
            preferred_device=args.preferred_device,
            max_length=args.max_length,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
