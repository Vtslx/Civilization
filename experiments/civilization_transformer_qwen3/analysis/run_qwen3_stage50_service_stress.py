from __future__ import annotations

import argparse
import json

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage49_persistent_inference_service import DEFAULT_CENTROID_BUNDLE, DEFAULT_PACKAGE_MANIFEST
from .stage50_service_stability_stress import (
    DEFAULT_OUTPUT_DIR,
    Stage50StressConfig,
    run_stage50_fake_stress,
    run_stage50_real_stress,
)


def _controls(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage50 resident inference service stability stress")
    parser.add_argument("--smoke", action="store_true", help="run fake-runtime smoke stress")
    parser.add_argument("--real", action="store_true", help="run real Qwen resident stress")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--total-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--controls", default="full,no_memory_path,no_rule_path,adapter_disabled")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--package-manifest", default=str(DEFAULT_PACKAGE_MANIFEST))
    parser.add_argument("--centroid-bundle", default=str(DEFAULT_CENTROID_BUNDLE))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not args.smoke and not args.real:
        parser.error("one of --smoke or --real is required")
    default = Stage50StressConfig() if args.smoke else Stage50StressConfig(total_requests=6, concurrency=2, batch_size=1)
    config = Stage50StressConfig(
        total_requests=args.total_requests if args.total_requests is not None else default.total_requests,
        concurrency=args.concurrency if args.concurrency is not None else default.concurrency,
        batch_size=args.batch_size if args.batch_size is not None else default.batch_size,
        controls=_controls(args.controls),
        max_batch_size=max(default.max_batch_size, args.batch_size or default.batch_size),
        port=args.port,
    )
    if args.smoke:
        summary = run_stage50_fake_stress(output_dir=args.output_dir, config=config)
    else:
        summary = run_stage50_real_stress(
            output_dir=args.output_dir,
            config=config,
            package_manifest=args.package_manifest,
            centroid_bundle=args.centroid_bundle,
            model_path=args.model_path,
            preferred_device=args.preferred_device,
            max_length=args.max_length,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
