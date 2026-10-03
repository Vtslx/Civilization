from __future__ import annotations

import argparse
import json

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage49_persistent_inference_service import DEFAULT_CENTROID_BUNDLE, DEFAULT_PACKAGE_MANIFEST
from .stage52_real_runtime_reload_stress import (
    DEFAULT_OUTPUT_DIR,
    Stage52ReloadStressConfig,
    run_stage52_fake_reload_stress,
    run_stage52_real_reload_stress,
)


def _controls(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage52 managed runtime reload stress")
    parser.add_argument("--smoke", action="store_true", help="run fake-runtime reload smoke")
    parser.add_argument("--real", action="store_true", help="run real Qwen runtime reload stress")
    parser.add_argument("--reload-rounds", type=int, default=1)
    parser.add_argument("--controls", default="full,no_memory_path,adapter_disabled")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--package-manifest", default=str(DEFAULT_PACKAGE_MANIFEST))
    parser.add_argument("--centroid-bundle", default=str(DEFAULT_CENTROID_BUNDLE))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if not args.smoke and not args.real:
        parser.error("one of --smoke or --real is required")
    config = Stage52ReloadStressConfig(reload_rounds=args.reload_rounds, controls=_controls(args.controls))
    if args.smoke:
        summary = run_stage52_fake_reload_stress(output_dir=args.output_dir, config=config)
    else:
        summary = run_stage52_real_reload_stress(
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
