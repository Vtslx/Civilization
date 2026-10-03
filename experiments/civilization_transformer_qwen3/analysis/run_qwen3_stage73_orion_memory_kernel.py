from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage73_orion_memory_kernel import DEFAULT_OUTPUT_DIR, run_stage73_orion_memory_kernel_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage73 Orion multi-system memory kernel smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if not args.smoke:
        parser.error("Stage73 currently supports --smoke only")

    summary = run_stage73_orion_memory_kernel_smoke(output_dir=Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
