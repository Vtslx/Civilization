from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage74_orion_memory_service import (
    DEFAULT_OUTPUT_DIR,
    build_stage74_real_service,
    run_stage74_orion_memory_service_smoke,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage74 Orion memory service")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preferred-device", default="cuda")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    if args.smoke:
        summary = run_stage74_orion_memory_service_smoke(output_dir=Path(args.output_dir))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    service = build_stage74_real_service(
        port=args.port,
        preferred_device=args.preferred_device,
        model_path=args.model_path,
    )
    service.config = service.config.__class__(
        host=args.host,
        port=args.port,
        max_batch_size=service.config.max_batch_size,
        batch_window_ms=service.config.batch_window_ms,
        request_timeout_seconds=service.config.request_timeout_seconds,
        default_controls=service.config.default_controls,
        limits=service.config.limits,
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
