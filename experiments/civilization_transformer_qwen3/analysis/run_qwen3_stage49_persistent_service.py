from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage49_persistent_inference_service import (
    DEFAULT_CENTROID_BUNDLE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PACKAGE_MANIFEST,
    Stage49ServiceConfig,
    build_stage49_real_service,
    run_stage49_service_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage49 persistent Qwen3 Civilization inference service")
    parser.add_argument("--serve", action="store_true", help="start the real resident HTTP service")
    parser.add_argument("--smoke", action="store_true", help="run fake-runtime service smoke")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--package-manifest", default=str(DEFAULT_PACKAGE_MANIFEST))
    parser.add_argument("--centroid-bundle", default=str(DEFAULT_CENTROID_BUNDLE))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--batch-window-ms", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.smoke:
        summary = run_stage49_service_smoke(output_dir=args.output_dir, port=0)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not args.serve:
        parser.error("one of --serve or --smoke is required")
    service = build_stage49_real_service(
        package_manifest=args.package_manifest,
        centroid_bundle=args.centroid_bundle,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
        max_length=args.max_length,
        config=Stage49ServiceConfig(
            host=args.host,
            port=args.port,
            max_batch_size=args.max_batch_size,
            batch_window_ms=args.batch_window_ms,
        ),
    )
    print(
        json.dumps(
            {
                "stage": "stage49_persistent_inference_service",
                "status": "starting",
                "host": args.host,
                "port": args.port,
                "package_manifest": args.package_manifest,
                "centroid_bundle": args.centroid_bundle,
                "preferred_device": args.preferred_device,
                "max_length": args.max_length,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
