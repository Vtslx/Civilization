from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage48_production_jsonl_inference import (
    DEFAULT_OUTPUT_DIR,
    Stage48Limits,
    build_stage48_runtime,
    run_stage48_jsonl_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production-style Qwen3 Civilization JSONL inference")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "responses.jsonl"))
    parser.add_argument("--summary", default=str(DEFAULT_OUTPUT_DIR / "summary.json"))
    parser.add_argument("--package-manifest", default="artifacts/civilization/stage45_adapter_package/package_manifest.json")
    parser.add_argument("--centroid-bundle", default="artifacts/civilization/stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--controls", default="full")
    parser.add_argument("--max-requests", type=int, default=1000)
    args = parser.parse_args()
    controls = tuple(value for value in args.controls.split(",") if value)
    runtime = build_stage48_runtime(
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
    )
    summary = run_stage48_jsonl_inference(
        runtime=runtime,
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        default_controls=controls,
        limits=Stage48Limits(max_requests=args.max_requests),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
