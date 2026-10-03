from __future__ import annotations

import argparse
import json

from .stage75_orion_adapter_diagnostic import DEFAULT_OUTPUT_DIR, run_stage75_orion_adapter_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage75 Orion adapter-context diagnostic")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default="/home/yike/AoNeb-01/Models/Qwen3-0.6B")
    parser.add_argument("--preferred-device", default="cuda")
    args = parser.parse_args()
    summary = run_stage75_orion_adapter_diagnostic(
        output_dir=args.output_dir,
        model_path=args.model_path,
        preferred_device=args.preferred_device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
