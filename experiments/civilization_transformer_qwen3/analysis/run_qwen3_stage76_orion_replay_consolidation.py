from __future__ import annotations

import argparse
import json

from .stage76_orion_replay_consolidation import DEFAULT_OUTPUT_DIR, run_stage76_orion_replay_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage76 Orion replay/consolidation smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Stage76 currently supports --smoke only")
    print(json.dumps(run_stage76_orion_replay_smoke(output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
