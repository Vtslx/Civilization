from __future__ import annotations

import argparse
import json

from .stage53_concurrent_reload_stress import (
    DEFAULT_OUTPUT_DIR,
    Stage53ConcurrentReloadConfig,
    run_stage53_fake_concurrent_reload_stress,
    run_stage53_real_concurrent_reload_stress,
)


def _controls(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage53 concurrent reload stress")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--pre-reload-requests", type=int, default=None)
    parser.add_argument("--concurrent-requests", type=int, default=None)
    parser.add_argument("--post-reload-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--controls", default="full,no_memory_path,adapter_disabled")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not args.smoke and not args.real:
        parser.error("one of --smoke or --real is required")
    default = Stage53ConcurrentReloadConfig()
    if args.real:
        default = Stage53ConcurrentReloadConfig(pre_reload_requests=1, concurrent_requests=3, post_reload_requests=1)
    config = Stage53ConcurrentReloadConfig(
        pre_reload_requests=args.pre_reload_requests if args.pre_reload_requests is not None else default.pre_reload_requests,
        concurrent_requests=args.concurrent_requests if args.concurrent_requests is not None else default.concurrent_requests,
        post_reload_requests=args.post_reload_requests if args.post_reload_requests is not None else default.post_reload_requests,
        concurrency=args.concurrency if args.concurrency is not None else default.concurrency,
        controls=_controls(args.controls),
    )
    if args.smoke:
        summary = run_stage53_fake_concurrent_reload_stress(output_dir=args.output_dir, config=config)
    else:
        summary = run_stage53_real_concurrent_reload_stress(
            output_dir=args.output_dir,
            config=config,
            preferred_device=args.preferred_device,
            max_length=args.max_length,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
