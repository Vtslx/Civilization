from __future__ import annotations

import argparse
import json

from .stage51_runtime_management import DEFAULT_OUTPUT_DIR, run_stage51_management_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage51 runtime management smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("--smoke is required for Stage51 runner")
    summary = run_stage51_management_smoke(output_dir=args.output_dir, port=args.port)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
